import numpy as np
from nltk import sent_tokenize, word_tokenize

class Shuffler(object):
    def __init__(self, config,):
        super().__init__()
        self.config = config
    
    def shuffle_passage_list(self, passage_list, selected_list=None):
        selected_list = selected_list if selected_list else [0] * len(passage_list)

        gt_pos = np.argmax(selected_list)
        selected_list.pop(gt_pos)
        gt_doc = passage_list.pop(gt_pos)

        if passage_list and selected_list:
            combined = list(zip(passage_list, selected_list))
            shuffle_result = self._shuffle_list(combined, **self.config.get("paragraph"))
            passage_list, selected_list = zip(*shuffle_result)

            passage_list, selected_list = map(list, [passage_list, selected_list])
            assert len(passage_list) == len(selected_list)
        
        result_list = []
        for idx, (passage, selected) in enumerate(zip(passage_list, selected_list)):            
            result_list.append(self._shuffle_passage(passage, selected))
        
        rnd = np.random.uniform(0, 1) 
        result_list.insert(int(rnd * len(result_list)), gt_doc)
        selected_list.insert(int(rnd * len(selected_list)), 1)
        
        return result_list, selected_list
        
    def _shuffle_list(self, list_to_shuffle, shuffle_degree=0.95, drop_ratio=0.05, duplicate_ratio=0.2):
        assert isinstance(list_to_shuffle, list)
        
        shuffle_num = int(len(list_to_shuffle) * shuffle_degree)
        drop_num = int(len(list_to_shuffle) * drop_ratio)
        duplicate_num = int(len(list_to_shuffle) * duplicate_ratio)
        
        shuffle_chosen = self._list_sample(list_to_shuffle, len(list_to_shuffle) - drop_num, replace=False)
        shuffle_chosen_ranked = []
        for item in list_to_shuffle:
            if item in shuffle_chosen:
                shuffle_chosen_ranked.append(item)
        
        list_to_shuffle = shuffle_chosen_ranked[:]

        duplicate_part = self._list_sample(list_to_shuffle, duplicate_num, replace=len(list_to_shuffle) < duplicate_num)
        
        shuffle_part = self._list_sample(list_to_shuffle, shuffle_num, replace=False)
        np.random.shuffle(shuffle_part)
        remain_part = [item for item in list_to_shuffle if item not in shuffle_part]

        shuffled_list = shuffle_part + duplicate_part
        shuffled_list = self._list_sample(shuffled_list, len(shuffled_list), replace=False)
        shuffled_list += remain_part
        
        return shuffled_list

    def _list_sample(self, list_to_sample, sample_num, replace):
        sample_num = len(list_to_sample) if sample_num > len(list_to_sample) else sample_num
        indices = np.random.choice(range(len(list_to_sample)), sample_num, replace=replace)

        return [list_to_sample[idx] for idx in indices]
    
    def _shuffle_passage(self, passage, selected=False):
        sentences = sent_tokenize(passage)
        passage_config = self.config.get("passage").copy()
        if selected:
            passage_config["drop_ratio"] = 0.
        result = self._shuffle_list(sentences, **passage_config)
        result = list(map(lambda sentence: self._shuffle_sentence(sentence, selected), result))
        return " ".join(result)
    
    def _shuffle_sentence(self, sentence, selected=False):
        words = word_tokenize(sentence)
        sentence_config = self.config.get("sentence").copy()
        if selected:
            sentence_config["drop_ratio"] = 0.
        result = self._shuffle_list(words, **sentence_config)
        return " ".join(result)
    

shuffler = Shuffler(None)