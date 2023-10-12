import numpy as np
import torch
from tqdm import tqdm
from transformers import RepetitionPenaltyLogitsProcessor
from transformers.generation.logits_process import EncoderRepetitionPenaltyLogitsProcessor

from kcd.classifier_guidance.ppl_mcts import pad_sequences_to_left, pad_sequences_to_left_states
from kcd.openai_module import OpenAIModel, OpenAIAPIParameters

# The maximum length the LM and/or the classifier can handle. 512 in case of BERT
MAX_SEQUENCE_LENGTH = 512


class OpenAILMStep:

    def __init__(self,
                 lm: OpenAIModel,
                 parameters: OpenAIAPIParameters,
                 classi,
                 tokenizer,
                 repetition_penalty=1.2,
                 knowledge_copy_penalty=1.0,
                 unused_token_id=2):
        self.lm = lm
        self.parameters = parameters
        self.tokenizer = tokenizer
        self.classi = classi
        self.repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
        self.knowledge_copy_penalty = knowledge_copy_penalty
        self.k_copy_processor = None
        self.unused_token_id = unused_token_id

        self.reference_ids = None
        self.token_usages = []  # for openai usage stat

    def reset(self):
        self.reference_ids = None
        self.k_copy_processor = None
        self.token_usages = []

    def parse_logp(self, input_texts):
        logprobs = []
        token_usages = []
        for t in input_texts:
            _, outputs = self.lm(t, self.parameters)
            token_usages.append(outputs['usage']['total_tokens'])
            logp = outputs['choices'][0]['logprobs']["top_logprobs"][0]  # dict
            full_logp = torch.full((self.tokenizer.vocab_size,), -torch.inf)
            for k, v in logp.items():
                full_logp[self.tokenizer.convert_tokens_to_ids(k)] = v
            logprobs.append(full_logp)
        logprobs = torch.stack(logprobs, dim=0)  # [B, V]
        if len(self.token_usages) == 0:
            self.token_usages = token_usages
        else:
            for i, u in enumerate(token_usages):
                self.token_usages[i] += u
        return logprobs

    @torch.inference_mode()
    def root_eval(self, original_input, labels, continuing=False):
        if 'knowledge_ids' in original_input:
            self.reference_ids = original_input['knowledge_ids']
            self.k_copy_processor = EncoderRepetitionPenaltyLogitsProcessor(
                self.knowledge_copy_penalty, self.reference_ids)

        # get logprob
        input_ids = original_input['input_ids']
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        logprobs = self.parse_logp(input_texts).to(input_ids.device)  # [B, V]

        # penalty
        prompt_masked_input_ids = torch.clone(input_ids)
        inverted_attention_mask = original_input["attention_mask"] == 0
        # penalizing an unused token
        prompt_masked_input_ids[inverted_attention_mask] = self.unused_token_id
        priors = self.repetition_penalty(prompt_masked_input_ids, logprobs)
        if self.k_copy_processor is not None:
            priors = self.k_copy_processor(prompt_masked_input_ids, priors)

        priors = torch.exp(priors)  # logprob -> prob

        # Use of our discriminator to get values
        output, logits = self.classi(input_ids=original_input.input_ids,
                                     attention_mask=original_input.attention_mask,
                                     use_cache=True)
        if continuing:
            values = torch.softmax(logits, dim=-1)[labels.bool()].cpu()
        else:
            # no generation yet; set values to 0
            values = torch.zeros((labels.shape[0],), dtype=torch.float).cpu()
        classi_past_key_values = output.past_key_values

        return priors, values, classi_past_key_values

    @torch.inference_mode()
    def node_eval(self, classi_states, original_token_ids, original_attention_masks, labels):
        """
        *_states: decoder states (past key values cache)
        original_token_ids: decoder token ids being generated
        original_attention_masks: decoder attention masks
        """
        # get logprob
        input_texts = self.tokenizer.batch_decode(original_token_ids, skip_special_tokens=True)
        logprobs = self.parse_logp(input_texts).to(original_token_ids.device)  # [B, V]

        # repetition penalty
        prompt_masked_input_ids = torch.clone(original_token_ids)
        inverted_attention_mask = original_attention_masks == 0
        # penalizing an unused token
        prompt_masked_input_ids[inverted_attention_mask] = self.unused_token_id
        priors = self.repetition_penalty(prompt_masked_input_ids, logprobs)
        if self.k_copy_processor is not None:
            priors = self.k_copy_processor(prompt_masked_input_ids, priors)

        priors = torch.exp(priors)  # logprob -> prob

        # Use of our discriminator to get values
        model_inputs = self.classi.prepare_inputs_for_generation(
            input_ids=original_token_ids,
            attention_mask=original_attention_masks,
            past_key_values=classi_states,
            use_cache=True)
        output, logits = self.classi(**model_inputs, return_dict=True)
        classi_past_key_values = output.past_key_values
        values = torch.softmax(logits, dim=-1)[labels.bool()].cpu()
        return priors, values, classi_past_key_values


class OpenAIMCTS:

    def __init__(self,
                 args,
                 tokenizer,
                 lm,
                 openai_parameters,
                 classi,
                 batch_size=1,
                 top_k=6,
                 num_labels=2,
                 unused_token_id=2,
                 device='cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_simulations = args.num_simulations
        self._num_actions = len(tokenizer) + 1
        self._num_sparse_actions = min(top_k, self._num_actions)  # TODO
        self._pb_c_init = args.c
        self.alpha = args.alpha

        self.lm_step = OpenAILMStep(lm,
                                    openai_parameters,
                                    classi,
                                    tokenizer,
                                    repetition_penalty=args.penalty,
                                    knowledge_copy_penalty=args.knowledge_copy_penalty,
                                    unused_token_id=unused_token_id)
        self._adaptive_min_values = np.zeros(self._batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(self._batch_size, dtype=np.float32)
        self._labels = torch.zeros((self._batch_size, num_labels),
                                   dtype=torch.uint8,
                                   device=self.device)

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = args.num_simulations + 1
        batch_node = (self._batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (self._batch_size, num_nodes, self._num_sparse_actions)  # (B, )
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_probas = np.zeros((batch_size, num_nodes, self._num_sparse_actions),
                                         dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._classi_states = {}
        self._classi_cross_states = None
        self._original_token_ids = {}
        self._original_attention_mask = {}
        self._batch_range = np.arange(self._batch_size)
        self._reset_tree()

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_probas.fill(0.0)
        self._children_visits.fill(0)
        self._classi_states = {}
        self._classi_cross_states = None
        self._original_token_ids = {}  # Indexed by tuples (batch index, node index)
        self._original_attention_mask = {}
        self.lm_step.reset()

    def set_labels(self, labels):
        self._labels = labels

    def search(self, original_input, tokens_to_generate=98, continuing=False):
        self._reset_tree()

        # Evaluate the root.
        prior, values, classi_states = self.lm_step.root_eval(original_input,
                                                              self._labels,
                                                              continuing=continuing)
        if isinstance(values, tuple):
            # next_values: [B, V]
            values, child_prob = values
        else:
            child_prob = None
        decoder_token_ids = original_input.input_ids
        decoder_attention = original_input.attention_mask
        self._adaptive_min_values = 1
        self._adaptive_max_values = 1 + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, values, child_prob, decoder_token_ids,
                         decoder_attention, classi_states,
                         np.full(self._batch_size, False, dtype=bool))

        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        existing_nodes = 0
        tokens_pbar = tqdm(total=tokens_to_generate, desc="Tokens generated")
        for i in range(tokens_to_generate):
            # build MCTS for decoding next step
            for sim in range(self._num_simulations):
                node_indices, actions = self.simulate()
                next_node_index = sim + 1 + existing_nodes  # root is 0, therefore we offset by 1.
                self.expand(node_indices, actions, next_node_index)
                leaf_indices.fill(next_node_index)
                self.backward(leaf_indices)
            # compute tree stats
            visit_counts, _ = self.dense_visit_counts()
            existing_nodes = np.amax(visit_counts)
            # Create new tree with selected node as root
            num_nodes = self._num_simulations + existing_nodes + 1
            batch_node = (self._batch_size, num_nodes)
            temp_visit_counts = np.zeros(batch_node, dtype=np.int32)
            temp_values = np.zeros(batch_node, dtype=np.float32)
            temp_likelihoods = np.zeros(batch_node, dtype=np.float32)
            temp_raw_values = np.zeros(batch_node, dtype=np.float32)
            temp_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_action_from_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_depth = np.zeros(batch_node, dtype=np.int32)
            temp_is_terminal = np.full(batch_node, False, dtype=bool)
            batch_node_action = (self._batch_size, num_nodes, self._num_sparse_actions)  # (B, )
            temp_topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
            temp_children_index = np.full(batch_node_action, -1, dtype=np.int32)
            temp_children_prior = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_probas = np.zeros((self._batch_size, num_nodes, self._num_sparse_actions),
                                            dtype=np.float32)
            temp_children_values = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_visits = np.zeros(batch_node_action, dtype=np.int32)
            temp_classi_states = {}
            temp_original_token_ids = {}  # Indexed by tuples (batch index, node index)
            temp_original_attention_mask = {}

            # select the next token as the next root
            for b, new_root_action in enumerate(np.argmax(visit_counts, axis=1)):
                new_root_id = self._children_index[b, 0, new_root_action]
                new_node_id = 1
                old_to_new_id = {new_root_id: 0}
                children_to_explore = self._children_index[b, new_root_id][
                    self._children_index[b, new_root_id] != -1].tolist()
                while (len(children_to_explore) > 0):
                    child_id = children_to_explore.pop(0)
                    old_to_new_id[child_id] = new_node_id
                    children_to_explore += self._children_index[b, child_id][
                        self._children_index[b, child_id] != -1].tolist()
                    new_node_id += 1
                # update stats
                for old_id, new_id in old_to_new_id.items():
                    if (new_id != 0):
                        temp_parents[b, new_id] = old_to_new_id[self._parents[b, old_id]]
                        temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    for i, children in enumerate(self._children_index[b, old_id]):
                        if (children != -1):
                            temp_children_index[b, new_id, i] = old_to_new_id[children]
                    temp_visit_counts[b, new_id] = self._visit_counts[b, old_id]
                    temp_values[b, new_id] = self._values[b, old_id]
                    temp_likelihoods[b, new_id] = self._likelihoods[b, old_id]
                    temp_raw_values[b, new_id] = self._raw_values[b, old_id]

                    temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    temp_depth[b, new_id] = self._depth[b, old_id] - 1
                    temp_is_terminal[b, new_id] = self._is_terminal[b, old_id]

                    temp_topk_mapping[b, new_id] = self._topk_mapping[b, old_id]
                    temp_children_prior[b, new_id] = self._children_prior[b, old_id]
                    temp_children_probas[b, new_id] = self._children_probas[b, old_id]
                    temp_children_values[b, new_id] = self._children_values[b, old_id]
                    temp_children_visits[b, new_id] = self._children_visits[b, old_id]

                    temp_classi_states[(b, new_id)] = self._classi_states[(b, old_id)]
                    temp_original_token_ids[(b, new_id)] = self._original_token_ids[(b, old_id)]
                    temp_original_attention_mask[(b,
                                                  new_id)] = self._original_attention_mask[(b,
                                                                                            old_id)]

                temp_classi_states[(b, 0)] = torch.cat(
                    (self._classi_states[(b, 0)], self._classi_states[(b, new_root_id)]), 3)

            self._num_nodes = num_nodes
            self._visit_counts = temp_visit_counts
            self._values = temp_values
            self._likelihoods = temp_likelihoods
            self._raw_values = temp_raw_values
            self._parents = temp_parents
            self._action_from_parents = temp_action_from_parents
            # The 0-indexed depth of the node. The root is the only 0-depth node.
            # The depth of node i, is the depth of its parent + 1.
            self._depth = temp_depth
            self._is_terminal = temp_is_terminal
            self._topk_mapping = temp_topk_mapping
            self._children_index = temp_children_index
            self._children_prior = temp_children_prior
            self._children_probas = temp_children_probas
            self._children_values = temp_children_values
            self._children_visits = temp_children_visits
            self._original_token_ids = temp_original_token_ids
            self._original_attention_mask = temp_original_attention_mask
            self._classi_states = temp_classi_states
            tokens_pbar.update(1)
            # If every sequences is finished, stop
            if (self._is_terminal[:, 0].all()):
                break
        all_decoded = []
        all_ids = []
        for b in range(self._batch_size):
            decoded = self.tokenizer.decode(self._original_token_ids[(b, 0)],
                                            skip_special_tokens=False,
                                            clean_up_tokenization_spaces=True)
            decoded = decoded.replace("\n", "")
            decoded = decoded.replace(f" {self.tokenizer.pad_token}", "")
            decoded = decoded.replace(f"{self.tokenizer.pad_token} ", "")
            decoded = decoded.replace(self.tokenizer.eos_token, "")
            all_decoded.append(decoded)
            all_ids.append(self._original_token_ids[(b, 0)])
        return all_decoded, all_ids

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None],
                           self._topk_mapping[:, root_index, :]] = root_visit_counts
        return root_visit_counts, dense_visit_counts

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)

    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :]  # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :]  # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :]  # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices]  # (B)
        node_policy_score = np.sqrt(
            node_visits[:,
                        None]) * self._pb_c_init * node_children_prior / (node_children_visits + 1)
        # (B, A)

        node_value_score = node_children_values

        node_uct_score = node_value_score + node_policy_score  # (B, A)
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    # return state
    def get_states_from_node(self, b, n, d, states):
        state_array = [None] * d
        state_array[d - 1] = states[(b, n)]
        while n != 0:
            n = self._parents[(b, n)]
            d -= 1
            state_array[d - 1] = states[(b, n)]
        return torch.cat(state_array, 3)

    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""
        # Retrieve token ids and masks for nodes to be evaluated.
        original_tokens_ids = pad_sequences_to_left(
            [self._original_token_ids[(b, n)] for b, n in enumerate(node_indices)], True,
            self.tokenizer.pad_token_id)
        original_attention_masks = pad_sequences_to_left(
            [self._original_attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)
        depths = torch.tensor([self._depth[(b, n)] + 1 for b, n in enumerate(node_indices)],
                              device=self.device)
        children_priors = np.array(
            [self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        previous_values = np.array([self._values[(b, n)] for b, n in enumerate(node_indices)])
        previous_node_is_terminal = self._is_terminal[self._batch_range,
                                                      node_indices[self._batch_range]]  # (B)

        classi_states_tensor = pad_sequences_to_left_states(
            [
                self.get_states_from_node(b, n, depths[b].item(), self._classi_states)
                for b, n in enumerate(node_indices)
            ],
            0,
            max_len=len(original_tokens_ids[0]),
            device=self.device,
        )
        if (len(original_tokens_ids[0]) >= MAX_SEQUENCE_LENGTH):
            previous_node_is_terminal[
                torch.sum(original_attention_masks, axis=1).cpu() >= MAX_SEQUENCE_LENGTH] = True
            original_tokens_ids = original_tokens_ids[:, -(MAX_SEQUENCE_LENGTH - 1):]
            original_attention_masks = original_attention_masks[:, -(MAX_SEQUENCE_LENGTH - 1):]
            classi_states_tensor = classi_states_tensor[:, :, :, :, -(MAX_SEQUENCE_LENGTH - 1):]

        classi_states = tuple(tuple(head) for head in classi_states_tensor)

        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        dense_actions[previous_node_is_terminal] = self.tokenizer.pad_token_id
        # Add actions to list of tokens and extend attention mask by 1
        original_tokens_ids = torch.cat(
            (original_tokens_ids, torch.unsqueeze(
                torch.LongTensor(dense_actions).to(self.device), 1)),
            dim=1)
        original_attention_masks = torch.cat(
            (original_attention_masks,
             torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device=self.device),
                             1)),
            dim=1)

        # Check if expanded nodes are terminal
        expanded_node_is_terminal = np.logical_or((dense_actions == self.tokenizer.pad_token_id),
                                                  previous_node_is_terminal)
        # Evaluate nodes.
        (prior, next_values,
         classi_states) = self.lm_step.node_eval(classi_states, original_tokens_ids,
                                                 original_attention_masks, self._labels)

        child_prob = None
        values = next_values
        values.numpy()[previous_node_is_terminal] = previous_values[previous_node_is_terminal]

        # Store unpaded version of inputs to save space
        original_attention_masks = [
            torch.cat((self._original_attention_mask[(b, n)],
                       torch.ones(1, dtype=torch.long, device=self.device)),
                      dim=0) for b, n in enumerate(node_indices)
        ]
        original_tokens_ids = [
            torch.cat((self._original_token_ids[(b, n)], torch.LongTensor([dense_actions[b]]).to(
                self.device)),
                      dim=0) for b, n in enumerate(node_indices)
        ]

        # Create the new nodes.
        self.create_node(next_node_index, prior, likelihoods * children_priors, values, child_prob,
                         original_tokens_ids, original_attention_masks, classi_states,
                         expanded_node_is_terminal)

        # Update the min and max values arrays
        self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)

        # Update tree topology.
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

    def create_node(self, node_index, prior, likelihoods, values, child_prob, original_tokens_ids,
                    original_attention_masks, classi_states, expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        # prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions,
        #                                      axis=-1)[:, -self._num_sparse_actions:]
        # prior = prior[self._batch_range[:, None], prior_topk_indices]  # (B, A)
        # NOTE: prior must be on GPU when using 8bit quantization to perform topk.
        prior, prior_topk_indices = prior.topk(self._num_sparse_actions, dim=-1)
        prior, prior_topk_indices = prior.cpu().numpy(), prior_topk_indices.cpu().numpy()
        if child_prob is not None:
            child_prob = child_prob[self._batch_range[:, None], prior_topk_indices]  # [B, A]
        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        if child_prob is not None:
            self._children_probas[:, node_index, :] = child_prob
        self._likelihoods[:, node_index] = likelihoods
        # raw_values = values**(self.alpha) * likelihoods**(1-self.alpha)
        raw_values = values
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal
        # Decoder: States has shape [12 (nhead), 2(K, V), batch_size, 12(nlayer), seq_len, 64]
        # Enc-Dec: States has shape [12 (nhead), 4(K, V * 2), batch_size, 12(nlayer), seq_len (dec, enc), 64]
        classi_key_value_tensor = torch.stack(
            [torch.stack(list(cls_kv_states[:2]), dim=0) for cls_kv_states in classi_states], dim=0)

        # If root, store the whole states
        if (node_index == 0):
            for b in range(len(original_tokens_ids)):
                self._classi_states[(b, node_index)] = torch.clone(classi_key_value_tensor[:, :, b])
        # Else just store the additional token hidden states to save space
        else:
            for b in range(len(original_tokens_ids)):
                self._classi_states[(b, node_index)] = torch.clone(classi_key_value_tensor[:, :,
                                                                                           b, :,
                                                                                           -1:])

        # Updates tokens ids
        for b, original_token_ids in enumerate(original_tokens_ids):
            self._original_token_ids[(b, node_index)] = original_token_ids

        # Updates attention masks
        for b, original_attention_mask in enumerate(original_attention_masks):
            self._original_attention_mask[(b, node_index)] = original_attention_mask

    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices  # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                return
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = not_root_mask * (
                self._values[self._batch_range, parents] *
                self._visit_counts[self._batch_range, parents] +
                leaf_values) / (self._visit_counts[self._batch_range, parents] +
                                1.0) + root_mask * self._values[self._batch_range, parents]

            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range,
                                                                     node_indices])
            self._children_values[
                self._batch_range, parents, actions] = not_root_mask * self._values[
                    self._batch_range, node_indices] + root_mask * self._children_values[
                        self._batch_range, parents, actions]
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            node_indices = parents
