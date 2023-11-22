import numpy as np
import pickle

from torch.utils.data import Dataset

from grid import (Distshift, RandomHorizontalLavaDistshift, RandomLavaDistshift, RandomLength1LavaDistshift, RandomLength2LavaDistshift, RandomVerticalLavaDistshift)


def make_unique_dataset(env, filename):
    start_positions = env.get_start_positions()
    actions = [0, 1, 2, 3]
    
    dataset = []
    for start_position in start_positions:
        for action in actions:
            obs, _ = env.reset(agent_start_pos=start_position)
            new_obs, reward, term, trun, _ = env.step(action)
            dataset.append({"image": obs, "action": action, "terminate": term, "reward": reward, "new_image": new_obs})
            
    with open(f"data/datasets/{filename}.pickle", 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)        


class DistshiftDataset(Dataset):
    def __init__(self, shift=False) -> None:
        filename = ("distshift2.pickle" if shift else "distshift.pickle")
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)

        flat_dataset = []
        start_img = dataset[0][0]["image"].copy()
        for traj in dataset:
            for i, sard in enumerate(traj):
                if i == (len(traj) - 1):
                    sard["new_image"] = start_img.copy()
                else:
                    sard["new_image"] = traj[i+1]["image"].copy()
                flat_dataset.append(sard)
        self.dataset = flat_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        sard = self.dataset[i]
        onehot_action = np.zeros(4)
        onehot_action[sard["action"]] = 1
        return sard["image"], onehot_action, sard["reward"], sard["terminate"], sard["new_image"]
    
    def concatenate(self, other):
        self.dataset += other.dataset
        

class UniqueDistshiftDataset(DistshiftDataset):
    def __init__(self, filenames, train=None, rand_mode="default") -> None:
        if isinstance(filenames, str):
            filenames = [filenames]
            
        self.dataset = []
        for filename in filenames:
            with open(f"data/datasets/{filename}.pickle", 'rb') as handle:
                self.dataset += pickle.load(handle)
        self.dataset = np.array(self.dataset)
        
        if train is not None:
            rng = np.random.default_rng(42)
            if rand_mode == "default":
                dataset_indices = np.arange(len(self.dataset))
                rng.shuffle(dataset_indices)
                
                if train is True:
                    dataset_indices = dataset_indices[:int(len(dataset_indices) * .8)]
                else:
                    dataset_indices = dataset_indices[int(len(dataset_indices) * .8):]
            elif rand_mode == "state":
                dataset_indices = np.arange(0, len(self.dataset), 4)
                rng.shuffle(dataset_indices)
                
                if train is True:
                    dataset_indices = dataset_indices[:int(len(dataset_indices) * .8)]
                else:
                    dataset_indices = dataset_indices[int(len(dataset_indices) * .8):]
                
                new_dataset_indices = []
                for i in dataset_indices:
                    new_dataset_indices += [i, i+1, i+2, i+3]
                dataset_indices = new_dataset_indices
            else:
                raise ValueError(f"Unrecognized {rand_mode=}")
                
            self.dataset = self.dataset[dataset_indices]


if __name__=="__main__":    
    make_unique_dataset(Distshift(shift=False), filename="distshift-v0")
    make_unique_dataset(Distshift(shift=True), filename="distshift-v1")
