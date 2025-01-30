import numpy as np
from tester.evaluation.data import BaseDataset, SequenceList, Sequence
from tester.utils.load_text import load_text


class RESOTDataset(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.base_path = dataset_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/ground truth.txt'.format(self.base_path, class_name, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        full_occlusion = 0
        out_of_view = 0

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)
        frames_path = '{}/{}/{}/Frames'.format(self.base_path, class_name, sequence_name)
        frames_list = ['{}/{:08d}.png'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        events_path = '{}/{}/{}/Events'.format(self.base_path, class_name, sequence_name)
        events_list = ['{}/{:08d}.npy'.format(events_path, event_number) for event_number in range(1, ground_truth_rect.shape[0] + 1)]


        target_class = class_name

        return Sequence(sequence_name, frames_list, events_list, 'rseot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['defocus-5',
                         'defocus-8',
                         'low_light-2',
                         'low_light-3',
                         'low_light-5',
                         'normal-5',
                         'occlusion-3',
                         'occlusion-5',
                         'occlusion-13',
                         'others-7',
                         ]
        return sequence_list