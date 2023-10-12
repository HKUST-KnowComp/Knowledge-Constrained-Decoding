"""
Taken from https://github.com/zhaoxlpku/KnowledGPT

Adjusted by Sehyun Choi, 2023
"""
import argparse
import os
import json


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'


def _first_val(dictionary):
    return list(dictionary.values())[0]


def _first_key(dictionary):
    return list(dictionary.keys())[0]


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (cand_title1
                    and cand_title1 in k_dict
                    and sentence in k_dict[cand_title1]):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def _parse_knowledge(obs, correct_first=False):
    if 'knowledge_parsed' in obs:
        # make a copy of the list to prevent the future padding step from
        # being destructive
        return list(obs['knowledge_parsed'])

    checked_sentence = '{} {} {}'.format(
        obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence']
    )
    # grab all the nonempty knowledge
    obs_know = [k.strip() for k in obs.get('knowledge', '').split('\n')]
    obs_know = [k for k in obs_know if k]

    # we wish the knowledge sentences to keep their original order
    # # we want the correct knowledge to always be in index 0
    if correct_first:
        try:
            i = obs_know.index(checked_sentence)
        except ValueError:
            # uh oh, couldn't find the sentence in the knowledge. This happens for
            # one or two examples in the training set. We can just artificially
            # put it back in
            i = 0
            obs_know[0] = checked_sentence
        obs_know[0], obs_know[i] = obs_know[i], obs_know[0]

    obs['knowledge_parsed'] = obs_know
    obs['checked_sentence_parsed'] = checked_sentence
    return obs['knowledge_parsed']


def len_episode(d):
    wizard_first = 'Wizard' in d['dialog'][0]['speaker']
    if wizard_first:
        return (len(d['dialog']) - 1) // 2
    return len(d['dialog']) // 2


def load_data(data_path):
    # 1. load from source file
    print('loading: {}'.format(data_path))
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. split into multiple turns
    examples = []
    for episode_idx, d in enumerate(data):
        for entry_idx in range(len_episode(d)):
            episode_done = entry_idx == (len_episode(d) - 1)

            wizard_first = 'Wizard' in d['dialog'][0]['speaker']
            idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

            # 2.1 get knowledge
            apprentice_ret_passages = wizard_ret_passages = {}

            if not wizard_first or idx != 0:
                apprentice_entry = d['dialog'][idx - 1]
                apprentice_ret_passages = apprentice_entry['retrieved_passages']
            if idx - 2 >= 0:
                wizard_prev_entry = d['dialog'][idx - 2]
                wizard_ret_passages = wizard_prev_entry['retrieved_passages']

            chosen_topic_passages = d['chosen_topic_passage']
            chosen_topic = d.get('chosen_topic', '')

            knowledge_dict = {chosen_topic: chosen_topic_passages}
            for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                for passage in ret_passes:
                    for k, v in passage.items():
                        if k not in knowledge_dict.keys():
                            knowledge_dict[k] = v

            # 2.2 get text
            if idx == 0:
                text = chosen_topic
            elif idx == 1:
                text = '{}\n{}'.format(chosen_topic, apprentice_entry['text'])
            else:
                text = apprentice_entry['text']

            # 2.3 get label
            wizard_entry = d['dialog'][idx]
            labels = [wizard_entry['text']]

            # 2.4 get label_candidates
            knowledge_str = ''
            for title, passage in knowledge_dict.items():
                for p in passage:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                    knowledge_str += cand + '\n'
            if not knowledge_str.startswith(TOKEN_NOCHOSEN):
                knowledge_str = (
                        TOKEN_NOCHOSEN
                        + ' '
                        + TOKEN_KNOWLEDGE
                        + ' '
                        + TOKEN_NOCHOSEN
                        + '\n'
                        + knowledge_str
                )

            # 2.5 get title and checked_sentences
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)

            examples.append({
                'text': text,
                'labels': labels,
                'chosen_topic': chosen_topic,
                'episode_done': episode_done,
                'knowledge': knowledge_str,
                'title': title,
                'checked_sentence': sentence,
            })
    print('loaded {} episodes with a total of {} examples'.format(episode_idx + 1, len(examples)))
    return examples


END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]  # acceptable ways to end a sentence


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def data_generator(in_file, correct_first=False, keep_last_n=99999):

    examples = load_data(in_file)
    observation = None

    history_strings = []
    users = []

    reset_on_next_update = False

    for i, ex in enumerate(examples):
        if i % 1000 == 0:
            print("Processing {} of {}; {:0.2f} percent done".format(
                i, len(examples), float(i) * 100.0 / float(len(examples))))

        if not observation or observation['episode_done']:
            last_reply = None
        else:
            last_reply = observation['labels'][0].lower()
            # last_reply = ' '.join(word_tokenize(last_reply))
            # last_reply = ' '.join(last_reply.split())
            last_reply = fix_missing_period(last_reply)
        observation = ex.copy()

        # 1. update the history using the observation
        if reset_on_next_update:
            history_strings = []
            users = []
            reset_on_next_update = False

        if last_reply is not None:
            history_strings.append(last_reply)
            users.append(1)

        if 'text' in observation and observation['text'] is not None:
            next_text = observation['text'].lower()
            # next_text = ' '.join(word_tokenize(next_text))
            # next_text = ' '.join(next_text.split())
            next_text = fix_missing_period(next_text)
            history_strings.append(next_text)
            users.append(0)

        if observation['episode_done']:
            reset_on_next_update = True

        # 2. parse history, label and knowledge
        # history = ' '.join(history_strings[-keep_last_n:])

        label = observation['labels'][0].lower()
        # label = ' '.join(word_tokenize(label))
        # label = ' '.join(label.split())

        knowledge = _parse_knowledge(observation, correct_first)
        knowledge = [k.lower() for k in knowledge]
        # knowledge = [' '.join(word_tokenize(k)) for k in knowledge]
        # knowledge = [' '.join(k.split()) for k in knowledge]

        yield (history_strings[-keep_last_n:], users[-keep_last_n:], label, knowledge)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_file', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    parser.add_argument('--keep_last_n', type=int, default=2)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for history, user, response, knowledge in data_generator(args.in_file, correct_first=True, keep_last_n=args.keep_last_n):
            f.write(
                json.dumps({
                    'history': history,
                    'user': user,
                    'response': response,
                    'knowledge': knowledge
                }) + '\n'
            )
