"""Reconstructed model inputs, with explicit legacy/strict feature loading modes."""
import logging
import os
import math

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader


def load_legacy_feature_pair(feature6_root, feature10_root, area, year, storm, timestamp):
    """Match the original TC3 loader, including lookup order and partial-load fallback.

    Only timestamp.npy is considered. No map pooling or suffixed-file discovery.
    A failed load leaves previously loaded values intact, as in the old shared try.
    """
    data_6, data_10 = np.zeros(6), np.zeros(10)
    if int(year) >= 2002:
        alternative = storm.lower().capitalize()
        try:
            path = os.path.join(feature6_root, area, year, storm, timestamp + '.npy')
            if not os.path.exists(path):
                path = os.path.join(feature6_root, area, year, alternative, timestamp + '.npy')
            if os.path.exists(path):
                data_6 = np.load(path)
            path = os.path.join(feature10_root, area, year, storm, timestamp + '.npy')
            if not os.path.exists(path):
                path = os.path.join(feature10_root, area, year, alternative, timestamp + '.npy')
            if os.path.exists(path):
                data_10 = np.load(path)
        except Exception:
            pass
    return data_6, data_10

# logger = logging.getLogger(__name__)


def env_data_processing(env_data):
    env_data_merge = {}
    obs_len = env_data[0]['wind'].shape[0]
    batch = len(env_data)
    for key in env_data[0]:
        if key == 'location':
            continue
        env_data_merge[key] = []
    for env_data_item in env_data:
        for key in env_data_item:
            if key == 'location':
                continue
            env_data_merge[key].append(torch.tensor(env_data_item[key]))

    for key in env_data_merge:
        env_data_merge[key] = torch.stack(env_data_merge[key],dim=0).reshape(batch,obs_len,-1).type(torch.float)

    return env_data_merge

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
     obs_date_mask, pred_date_mask,image_obs,image_pre,env_data,tyID) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    obs_traj_Me = torch.cat(obs_traj_Me, dim=0).permute(2, 0, 1)
    pred_traj_Me = torch.cat(pred_traj_gt_Me, dim=0).permute(2, 0, 1)
    obs_traj_rel_Me = torch.cat(obs_traj_rel_Me, dim=0).permute(2, 0, 1)
    pred_traj_rel_Me = torch.cat(pred_traj_gt_rel_Me, dim=0).permute(2, 0, 1)
    obs_date_mask = torch.cat(obs_date_mask, dim=0).permute(2, 0, 1)
    pred_date_mask = torch.cat(pred_date_mask, dim=0).permute(2, 0, 1)
    image_obs = torch.stack(image_obs, dim=0).permute(0,4,1,2,3)
    image_pre = torch.stack(image_pre, dim=0).permute(0,4,1,2,3)
    env_data = env_data_processing(env_data)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, obs_traj_Me, pred_traj_Me, obs_traj_rel_Me, pred_traj_rel_Me,
        obs_date_mask,pred_date_mask,image_obs,image_pre, env_data,tyID
    ]

    return tuple(out)




def read_file(_path, delim='\t'):
    data = []
    add = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            add.append(line[-2:])
            line = [float(i) for i in line[:-2]]
            data.append(line)
    return {'main':np.asarray(data),'addition':add}


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t',other_modal='gph', areas=None,year_filter=[],test_year=None
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        if areas is None:
            areas = ['WP']
        if test_year is None:
            test_year = '.txt'
        self.data_dir = data_dir
        mode = self.data_dir.get('feature_mode', 'legacy')
        if mode not in ('legacy', 'strict'):
            raise ValueError('Unknown feature_mode: ' + mode)
        if mode == 'legacy' and self.data_dir.get('feature_reduction', 'none') != 'none':
            raise ValueError('Legacy features do not use spatial_mean; use --feature-mode strict explicitly')
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.modal_name = other_modal
        self.areas = areas


        all_files = []
        for area in self.areas:
            area_files = os.listdir(os.path.join(self.data_dir['root'],area,self.data_dir['type']))
            # Optional stable ordering for portable checkpoint evaluation.
            if self.data_dir.get('sort_files', False):
                area_files.sort()
            area_files = [os.path.join(self.data_dir['root'],area,self.data_dir['type'], _path) for _path in area_files if test_year in _path]
            all_files = all_files+area_files

        # filter_files = []
        # if self.data_dir['type'] == 'test':
        #     year_filter = ['2020', '2021']
        # if len(year_filter) != 0:
        #     for file_i, file_name in enumerate(all_files):
        #         year_get = file_name.split('/')[-1][2:6]
        #         if year_get not in year_filter:
        #         # for year_f in year_filter:
        #         #     if year_f not in file_name:
        #             filter_files.append(file_name)
        #
        #     all_files = filter_files

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_date_mask = []
        loss_mask_list = []
        non_linear_ped = []
        tyID = []

        # 迭代读取文件夹中的txt文件
        for path in all_files:
            _,x = os.path.split(path)
            tyname = os.path.splitext(x)[0]
            data = read_file(path, delim)
            addinf = data['addition']
            data = data['main']
            # print(path)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                # 将txt文件中同一帧的，没有目标的坐标点保存在同一个frame_data的同一个index中
                frame_data.append(data[frame == data[:, 0], :])
            # 获取一个txt文件中的轨迹可以分割成多少个子轨迹，如第一个子轨迹：1,2,3...20，第二2,3,4，...21
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            # 迭代子轨迹
            for idx in range(0, num_sequences * self.skip + 1, skip):
                # axis=0  照着row的方向叠在一起
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                # peds_in_curr_seq  存储当前子序列的目标id，   ty的话   应该就是1
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                # 获取时间信息
                curr_date_mask = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    curr_ped_date_mask = [x[0] for x in addinf[idx:idx+pred_len+obs_len]]
                    curr_ped_date_mask = self.embed_time(curr_ped_date_mask)
                    # Make coordinates relative
                    # rel_curr_ped_seq 存相邻两个坐标点中间的差  后-前
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_date_mask[_idx, :, pad_front:pad_end] = curr_ped_date_mask
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # if num_peds_considered > min_ped: 源码---
                # - min_ped: Minimum number of pedestrians that should be in a seqeunce
                # 最小的行人个数，1的话  应该是至少有一个  但是源码缺大于1，有点问题。。。
                # 但是因为台风基本上同个时间都只有一个，所以我改成>=1
                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_date_mask.append(curr_date_mask[:num_peds_considered])
                    tyID.append({'old':[tyname,idx],'new':addinf[idx+self.obs_len-1],
                                 'tydate':[x[0] for x in addinf[idx:idx+pred_len+obs_len]]})

        self.num_seq = len(seq_list)
        if not seq_list:
            raise ValueError("No complete observation/forecast windows in the selected data")
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_date_mask = np.concatenate(seq_list_date_mask, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :2, self.obs_len:]).type(torch.float)
        self.obs_traj_Me = torch.from_numpy(
            seq_list[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_Me = torch.from_numpy(
            seq_list[:, 2:, self.obs_len:]).type(torch.float)
        self.obs_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, :self.obs_len]).type(torch.float)
        self.pred_traj_rel_Me = torch.from_numpy(
            seq_list_rel[:, 2:, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        self.obs_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, :self.obs_len]).type(torch.float)
        self.pred_date_mask = torch.from_numpy(
            seq_list_date_mask[:, :, self.obs_len:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        # seq_start_end 一个元组的索引代表的是同子序列中不同id的轨迹
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.tyID = tyID

    def __len__(self):
        return self.num_seq

    def embed_time(self,date_list):
        data_embed = []
        for date in date_list:
            year = (float(date[:4]) - 1949) / (2019 - 1949) - 0.5
            month = (float(date[4:6]) - 1) / 11.0 - 0.5
            day = (float(date[6:8]) - 1) / 30.0 - 0.5
            hour = float(date[8:10]) / 18 - 0.5
            data_embed.append([year, month, day, hour])
        return np.array(data_embed).transpose(1, 0)[np.newaxis, :, :]

    def transforms(self,img):
        # mean=np.array([111.2762205937308])
        # std = np.array([59.03717801611257])
        # return (img-mean)/std
        modal_range = {'gph': (44490.578125,58768.4486860389),
                      'mcc': (0,1),
                      '10v': (-48.2635498046875, 53.091079711914055),
                      '10u': (-49.08894348144531, 47.56512451171875),
                      'sst': (273,312),
                      '100v': 'data_test/wind_year_V100_centercrop',
                      '100u': 'data_test/wind_year_U100_centercrop',
                      }


        all_min,all_max = modal_range[self.modal_name]
        img = (img-all_min)/(all_max-all_min)
        img[img>1] = 1
        img[img<0] = 0
        return img

    def img_read(self, img_path):
        # Missing observed imagery is a data error, never an implicit zero input.
        if not os.path.isfile(img_path):
            raise FileNotFoundError(img_path)
        img = np.load(img_path, allow_pickle=False)
        if img.ndim != 2 or not np.isfinite(img).all():
            raise ValueError('Expected finite 2-D observed GPH at ' + img_path)
        img = cv2.resize(img.astype(np.float32), (64, 64))
        return self.transforms(img)[:, :, np.newaxis]

    # 文件: TCNM/data/trajectoriesWithMe_unet_training.py

    def get_img(self, tyid_dic):
        # 1. 解析基础信息
        tyname = tyid_dic['new'][1]
        tyname2 = tyname.lower().capitalize()
        tyname3 = tyname.upper()
        area = tyid_dic['old'][0][:2]
        
        # 提取年份，用于判断是否加载新数据
        year = tyid_dic['old'][0][2:6]
        year_int = int(year)
        
        tydate = tyid_dic['tydate']

        # 2. 读取基础环境数据 (Wind, Pressure 等)
        env_root = self.data_dir.get(
            'env_root',
            r'data_test/all_area_correct_location_includeunder15_2023',
        )
        env_data_obs = {}
        
        # 遍历观测时间步
        for env_date in tydate[:self.obs_len]:
            env_path = os.path.join(env_root, area, year, tyname, env_date + '.npy')
            if os.path.exists(env_path):
                env_data = np.load(env_path, allow_pickle=True).item()
            elif os.path.exists(os.path.join(env_root, area, year, tyname2, env_date + '.npy')):
                env_path = os.path.join(env_root, area, year, tyname2, env_date + '.npy')
                env_data = np.load(env_path, allow_pickle=True).item()
            else:
                env_path = os.path.join(env_root, area, year, tyname3, env_date + '.npy')
                env_data = np.load(env_path, allow_pickle=True).item()
            
            if len(env_data_obs) == 0:
                for key in env_data:
                    if key == 'location':
                        continue
                    env_data_obs[key] = [env_data[key]]
            else:
                for key in env_data:
                    if key == 'location':
                        continue
                    env_data_obs[key].append(env_data[key])
        
        # 处理基础环境数据中的缺失值 (-1)
        # =======================================================
        no_fu1 = {}
        for key in env_data_obs:
            # 1. 初始化标志位
            found_valid = False
            
            # 2. 尝试寻找序列中第一个非缺失值
            for data in env_data_obs[key]:
                is_valid = False
                if isinstance(data, np.ndarray):
                    # 如果是数组，只要不全为 -1 即视为有效（根据具体业务逻辑调整，这里假设全-1为缺失）
                    if not np.all(data == -1):
                        is_valid = True
                else:
                    # 如果是标量
                    if data != -1:
                        is_valid = True

                if is_valid:
                    no_fu1[key] = data
                    found_valid = True
                    break
            
            # 3. 如果整段序列都缺失 (found_valid 为 False)，填入对应维度的全0数据
            if not found_valid:
                # 根据 env_net 定义的维度进行初始化
                if key in ['history_direction24', 'history_direction12']:
                     no_fu1[key] = np.zeros(8)
                elif key == 'history_inte_change24':
                     no_fu1[key] = np.zeros(4)
                elif key == 'location_long':
                     no_fu1[key] = np.zeros(36)
                elif key in ['location_lat', 'month']:
                     no_fu1[key] = np.zeros(12)
                elif key == 'intensity_class':
                     no_fu1[key] = np.zeros(6)
                else:
                     # 其他标量特征 (wind, move_velocity 等) 填 0.0
                     no_fu1[key] = 0.0

        # 4. 执行填补操作
        for key in env_data_obs:
            for data_i, data in enumerate(env_data_obs[key]):
                is_missing = False
                if isinstance(data, np.ndarray):
                    if np.all(data == -1):
                        is_missing = True
                elif data == -1:
                    is_missing = True
                
                if is_missing:
                    env_data_obs[key][data_i] = no_fu1[key]
            
            # 转回 numpy 数组
            env_data_obs[key] = np.array(env_data_obs[key])

        # =======================================================
        # [NEW] 处理 Feature 6 和 10 (仅 2002-2023)
        # =======================================================
        # FIXME: 请修改为真实维度 (如标量填1, 向量填长度)
        dim_f6 = 6   
        dim_f10 = 10 
        
        # FIXME: 请修改为真实数据路径
        feature6_root = self.data_dir.get(
            'feature6_root', r'data_test/features6',
        )
        feature10_root = self.data_dir.get(
            'feature10_root', r'data_test/features10',
        )
        
        f6_list = []
        f10_list = []
        
        for env_date in tydate[:self.obs_len]:
            if self.data_dir.get('feature_mode', 'legacy') == 'legacy':
                data_6, data_10 = load_legacy_feature_pair(
                    feature6_root, feature10_root, area, year, tyname, env_date)
                f6_list.append(data_6)
                f10_list.append(data_10)
                continue
            # 初始化默认为 0
            data_6 = np.zeros(dim_f6)
            data_10 = np.zeros(dim_f10)

            if year_int >= 2002:
                def load_feature(root, width):
                    candidates = [os.path.join(root, area, year, name, env_date + suffix)
                                  for name in (tyname, tyname2, tyname3)
                                  for suffix in ('.npy', '_features%d.npy' % width)]
                    path = next((p for p in candidates if os.path.isfile(p)), None)
                    if path is None:
                        raise FileNotFoundError('Missing observed feature: ' + candidates[0])
                    value = np.load(path, allow_pickle=False)
                    if value.shape == (width,):
                        pass
                    elif value.ndim == 3 and value.shape[0] == width and self.data_dir.get('feature_reduction') == 'spatial_mean':
                        value = value.mean(axis=(1, 2))
                    else:
                        raise ValueError('Expected vector of size %d, got %s at %s. '
                                         'Map pooling requires explicit --feature-reduction spatial_mean.'
                                         % (width, value.shape, path))
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite feature at ' + path)
                    return value.astype(np.float32)
                data_6 = load_feature(feature6_root, dim_f6)
                data_10 = load_feature(feature10_root, dim_f10)

            f6_list.append(data_6)
            f10_list.append(data_10)

        # 加入到字典中
        env_data_obs['feature6'] = np.array(f6_list)
        env_data_obs['feature10'] = np.array(f10_list)
        # =======================================================

        # 3. 读取图像数据 (Image Obs & Pre) - 之前报错是因为缺了这段
        modal_path = {'gph':r'data_test/all_ocean_gph500_2023',
                      'mcc':'data_test/wind_year_MCC_centercrop',
                      '10v':'data_test/wind_year_V10_centercrop',
                      '10u':'data_test/wind_year_U10_centercrop',
                      'sst':'data_test/SST_year_centercrop',
                      '100v': 'data_test/wind_year_V100_centercrop',
                      '100u': 'data_test/wind_year_U100_centercrop',
                      }
        if self.data_dir.get('image_root') is not None:
            modal_path[self.modal_name] = self.data_dir['image_root']
        
        # 确定图像数据的路径
        if os.path.exists(os.path.join(modal_path[self.modal_name], area, year, tyname)):
            data_dir = os.path.join(modal_path[self.modal_name], area, year, tyname)
        elif os.path.exists(os.path.join(modal_path[self.modal_name], area, year, tyname2)):
            data_dir = os.path.join(modal_path[self.modal_name], area, year, tyname2)
        elif os.path.exists(os.path.join(modal_path[self.modal_name], area, year, tyname3)):
            data_dir = os.path.join(modal_path[self.modal_name], area, year, tyname3)
        else:
            # 路径不存在时的调试打印，可选
            # print(os.path.join(modal_path[self.modal_name], area, year, tyname))
            # 为了防止报错，可以设置一个默认路径或者抛出异常，这里假设路径大概率存在
            data_dir = os.path.join(modal_path[self.modal_name], area, year, tyname)

        image_obs = []
        image_pre = []
        obs_list = tydate[:self.obs_len]
        pre_list = tydate[self.obs_len:]

        # 加载观测图像序列
        for obs_date in obs_list:
            img_path = os.path.join(data_dir, obs_date + '.npy')
            img = self.img_read(img_path)
            image_obs.append(img)
            
        # 加载预测图像序列
        for pre_date in pre_list:
            if self.data_dir.get('observed_only', False):
                # Distillation/inference consumes only historical image inputs.
                image_pre.append(np.zeros((64, 64, 1), dtype=np.float32))
                continue
            img_path = os.path.join(data_dir, pre_date + '.npy')
            img = self.img_read(img_path)
            image_pre.append(img)
            
        # 转换为 Tensor
        image_obs = torch.tensor(np.array(image_obs), dtype=torch.float)
        image_pre = torch.tensor(np.array(image_pre), dtype=torch.float)
        
        # 4. 返回完整字典
        return {'obs': image_obs, 'pre': image_pre, 'env': env_data_obs}

    def __getitem__(self, index):
        # 迭代一次获取一个子序列的所有行人的轨迹，台风的话只有一个轨迹.
        # 这里的子轨迹只有20点，8或12作为先决条件，这样的话，就有一个假设就是：
        # 当前时刻的    将来轨迹只与近8或者12个点信息有关
        start, end = self.seq_start_end[index]
        image = self.get_img(self.tyID[start:end][0])
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.obs_traj_Me[start:end,:],self.pred_traj_Me[start:end,:],
            self.obs_traj_rel_Me[start:end, :], self.pred_traj_rel_Me[start:end, :],
            self.obs_date_mask[start:end, :],self.pred_date_mask[start:end, :],
            image['obs'],image['pre'],image['env'],
            self.tyID[start:end]
        ]
        return out



