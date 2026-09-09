# LINGO动作与无时间戳条件的构造

条件与监督分开。这里同时保留核验交互和broad动作的条件构造，便于检查K、事件、权限、接触以及路线摘要；不是通用多任务数据生成器。文本接口目前有零向量，不能声称自由文本已充分训练。完整Scene/SDF和人体适配器原文件见source目录。

下面是备份源码的逐字摘录，不是伪代码，也不是独立可执行模块。完整imports、辅助函数和校验仍在对应原文件中。

## make_condition

来源：[hsi/stage3_joint/data.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/data.py:185)，原文件第 185–271 行。

```python
def make_condition(arrays, record, k=1):
    """Create B=1 condition and separate GT duration/contact tensors."""
    motion = torch.as_tensor(arrays['motion'], dtype=torch.float32)
    joints = torch.as_tensor(arrays['joints'], dtype=torch.float32)
    regions = torch.as_tensor(arrays['regions'], dtype=torch.float32)
    h, total = int(record['history_frames']), len(motion)-int(record['history_frames'])
    event = int(record['establish_frame_1based'])
    travel_end = int(record['sit_frame_1based'])-1
    anchors = keypose_frames(total, event, k, travel_end)
    # Travel remains its own event even for K=1. Event order is an input;
    # actual source boundary frame numbers appear only in supervision.
    boundaries = sorted(set([travel_end]+anchors+[total]))
    durations = np.diff([0] + boundaries)
    m = len(boundaries)
    z = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    b = lambda *shape: torch.zeros(*shape, dtype=torch.bool)
    i = lambda *shape: torch.zeros(*shape, dtype=torch.long)
    poses = motion[[h+a-1 for a in anchors]][None]
    pose_joints = joints[[h+a-1 for a in anchors]][None]-poses[..., None, :3]
    active = b(1, k, 8)
    targets = poses[..., None, :3].expand(1, k, 8, 3).clone()
    normals = z(1, k, 8, 3)
    ids = i(1, k, 8)-1
    semantics = i(1, k)
    labels = torch.as_tensor(arrays['contacts'], dtype=torch.float32)
    known = torch.as_tensor(arrays['contact_known'], dtype=torch.bool)
    seat_target = torch.as_tensor(arrays['seat_target'], dtype=torch.float32)
    for p, a in enumerate(anchors):
        idx = h+a-1
        active[0, p] = (labels[idx] > .5) & known[idx]
        if a <= travel_end:
            semantics[0, p] = SEMANTIC_IDS['walk']
        elif a == event:
            semantics[0, p] = SEMANTIC_IDS['sit']
        else:
            semantics[0, p] = SEMANTIC_IDS['hold']
        for r in (0, 3, 4):
            if active[0, p, r]:
                targets[0, p, r] = seat_target if r == 0 else regions[idx, r]
                if r in (3, 4):
                    targets[0, p, r, 2] = 0.
                normals[0, p, r] = torch.as_tensor(arrays.get('seat_normal',[0.,0.,1.])) if r == 0 else torch.tensor([0.,0.,1.])
                ids[0, p, r] = 1 if r == 0 else 0
    slot_types = i(1, m)
    owners = i(1, m)
    slot_contacts = b(1, m, 8)
    for s, boundary in enumerate(boundaries):
        owner = next((p for p,a in enumerate(anchors) if a >= boundary),k-1)
        owners[0, s] = owner
        slot_types[0, s] = 0 if boundary <= travel_end else 1 if boundary == event else 2
        if slot_types[0, s] == 2:
            trustworthy_hold=torch.as_tensor(record.get('hold_contact_requirements',[True]*8),dtype=torch.bool)
            slot_contacts[0, s] = active[0, owner] & trustworthy_hold
    locked = torch.ones(1, k, 22, dtype=torch.bool)
    locked[..., list(ADJUSTABLE_JOINTS)] = False
    tolerance = z(1, k, 22)
    tolerance[..., list(ADJUSTABLE_JOINTS)] = np.deg2rad(5.)
    initial = (labels[h-1] > .5) & known[h-1]
    initial_ids = i(1, 8)-1
    initial_ids[0, initial] = 0
    if initial[0]:
        initial_ids[0, 0] = 1
    route = z(1, m, 8)
    # Input/history-to-goal geometric displacement only; no GT trajectory or
    # per-frame velocity supplied as route features.
    delta = poses[0, -1, :2]-motion[h-1, :2]
    route[..., :2] = delta
    route[..., 2] = delta.norm()
    route[..., 6] = seat_target[2]
    route[..., 7] = (slot_types != 0).float()
    c = SequenceCondition(history=motion[:h][None], history_mask=torch.ones(1,h,dtype=torch.bool),
        history_joints=joints[:h][None]-motion[:h][None,:,None,:3],
        initial_contact_active=initial[None], initial_contact_target_ids=initial_ids,
        keyposes=poses, keypose_mask=torch.ones(1,k,dtype=torch.bool), keypose_joints=pose_joints,
        keypose_text=z(1,k,512), semantic_ids=semantics,
        scene_points=torch.as_tensor(arrays['scene_points'],dtype=torch.float32)[None],
        scene_features=torch.as_tensor(arrays['scene_features'],dtype=torch.float32)[None],
        scene_mask=torch.ones(1,len(arrays['scene_points']),dtype=torch.bool),
        contact_targets=targets, contact_normals=normals, contact_active=active,
        contact_body_regions=i(1,k,8)-1, contact_target_ids=ids,
        locked_root=torch.ones(1,k,3,dtype=torch.bool), locked_joints=locked,
        root_tolerance=z(1,k,3), joint_tolerance=tolerance,
        slot_mask=torch.ones(1,m,dtype=torch.bool), slot_types=slot_types, slot_keyposes=owners,
        slot_contact_active=slot_contacts, slot_release_active=b(1,m,8),
        minimum_frames=torch.ones(1,m,dtype=torch.long), keypose_slots=torch.tensor([[boundaries.index(a) for a in anchors]]),
        route_features=route, total_frames=torch.tensor([total]), fps=15.).validate()
    return c, motion[h:][None], torch.as_tensor(durations,dtype=torch.long)[None], labels[h:][None], known[h:][None]
```

## make_motion_condition

来源：[hsi/stage3_joint/motion_data.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/motion_data.py:133)，原文件第 133–186 行。

```python
def make_motion_condition(arrays, record, scene_arrays, k=1):
    total, h = int(record['future_frames']), 2
    start = int(record['array_start']); stop = start+h+total
    origin = np.asarray(record['world_origin'], np.float32)
    motion = torch.from_numpy(np.array(arrays['motion_world'][start:stop], copy=True))
    motion[:, :3] -= torch.from_numpy(origin)
    joints_relative = torch.from_numpy(np.array(arrays['joints_relative'][start:stop], copy=True))
    regions = torch.from_numpy(np.array(arrays['regions_world'][start:stop], copy=True))-torch.from_numpy(origin)
    labels = torch.from_numpy(np.array(arrays['contacts'][start:stop], copy=True))
    known = torch.from_numpy(np.array(arrays['contact_known'][start:stop], copy=True))
    if len(motion) != h+total:
        raise ValueError('Indexed motion source too short')
    anchors = anchor_frames(total, k, record['case_id'])
    indices = [h+a-1 for a in anchors]
    poses = motion[indices][None]
    active = ((labels[indices] > .5) & known[indices])[None]
    targets = regions[indices][None].clone()
    # Only floor is geometrically certified here. Keep all furniture in SDF.
    targets[..., 3:5, 2] = -float(origin[2])
    normals = torch.zeros(1, k, 8, 3); normals[..., 2] = active.float()
    ids = torch.full((1, k, 8), -1, dtype=torch.long); ids[active] = 0
    count = k+1
    locked = torch.ones(1, k, 22, dtype=torch.bool); locked[..., list(ADJUSTABLE_JOINTS)] = False
    tolerance = torch.zeros(1, k, 22); tolerance[..., list(ADJUSTABLE_JOINTS)] = np.deg2rad(5.)
    slot_types = torch.zeros(1, count, dtype=torch.long)
    slot_active = torch.zeros(1, count, 8, dtype=torch.bool)
    for n in range(k):
        if active[0, n].any():
            slot_types[0, n] = 1  # only establishes at its internal anchor
            slot_active[0, n] = active[0, n]
    initial = (labels[h-1] > .5) & known[h-1]
    initial_ids = torch.full((1, 8), -1, dtype=torch.long); initial_ids[0, initial] = 0
    scene_points = torch.from_numpy(np.array(scene_arrays['mesh_points'], copy=True))-torch.from_numpy(origin)
    features = torch.zeros(len(scene_points), 4)
    features[:, :3] = torch.from_numpy(np.array(scene_arrays['mesh_normals'], copy=True))
    route = torch.zeros(1, count, 8)
    previous = torch.cat((motion[h-1:h, :3], poses[0, :-1, :3]), 0)
    delta = poses[0, :, :3]-previous
    route[0, :k, :2] = delta[:, :2]; route[0, :k, 2] = delta[:, :2].norm(dim=-1)
    c = SequenceCondition(history=motion[:h][None], history_mask=torch.ones(1, h, dtype=torch.bool),
        history_joints=joints_relative[:h][None], initial_contact_active=initial[None], initial_contact_target_ids=initial_ids,
        keyposes=poses, keypose_mask=torch.ones(1, k, dtype=torch.bool), keypose_joints=joints_relative[indices][None],
        keypose_text=torch.zeros(1, k, 512), semantic_ids=torch.full((1, k), record['semantic_id'], dtype=torch.long),
        scene_points=scene_points[None], scene_features=features[None], scene_mask=torch.ones(1, len(scene_points), dtype=torch.bool),
        contact_targets=targets, contact_normals=normals, contact_active=active,
        contact_body_regions=torch.full((1, k, 8), -1, dtype=torch.long), contact_target_ids=ids,
        locked_root=torch.ones(1, k, 3, dtype=torch.bool), locked_joints=locked,
        root_tolerance=torch.zeros(1, k, 3), joint_tolerance=tolerance,
        slot_mask=torch.ones(1, count, dtype=torch.bool), slot_types=slot_types,
        slot_keyposes=torch.tensor([list(range(k))+[k-1]]), slot_contact_active=slot_active,
        slot_release_active=torch.zeros(1, count, 8, dtype=torch.bool), minimum_frames=torch.ones(1, count, dtype=torch.long),
        keypose_slots=torch.arange(k)[None], route_features=route, total_frames=torch.tensor([total]), fps=15.).validate()
    durations = torch.tensor([np.diff([0]+anchors+[total]).tolist()], dtype=torch.long)
    return c, motion[h:][None], durations, labels[h:][None], known[h:][None]
```

## JointBatch

来源：[hsi/stage3_joint/data.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/data.py:307)，原文件第 307–316 行。

```python
@dataclass
class JointBatch:
    condition: SequenceCondition
    clean: torch.Tensor
    timing: torch.Tensor
    contacts: torch.Tensor
    contact_known: torch.Tensor
    body: MeshBody
    scene: Scene
    metadata: list[dict]
```

## JointCorpus

来源：[hsi/stage3_joint/corpus.py](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/stage3_core_20260909/source/hsi/stage3_joint/corpus.py:13)，原文件第 13–53 行。

```python
class JointCorpus:
    def __init__(self, interaction_root, body_asset, device="cpu", motion_root=None):
        interaction = JointDataset(interaction_root,body_asset,device=device)
        self.datasets = [interaction]
        if motion_root is not None:
            from .motion_data import MotionJointDataset
            self.datasets.append(MotionJointDataset(motion_root,body_asset,device=device))
        self.records, self.lookup = [], []
        self.train_indices, self.heldout_indices = [], []
        self.interaction_train_indices, self.motion_train_indices = [], []
        self.manifest = dict(interaction=interaction.manifest,
            motion=self.datasets[1].manifest if len(self.datasets)>1 else None)
        for source,dataset in enumerate(self.datasets):
            offset = len(self.records)
            self.records.extend(dataset.records)
            self.lookup.extend((source,i) for i in range(len(dataset)))
            train = [offset+int(i) for i in dataset.train_indices]
            self.train_indices.extend(train)
            self.heldout_indices.extend(offset+int(i) for i in dataset.heldout_indices)
            (self.interaction_train_indices if source==0 else self.motion_train_indices).extend(train)
        train_scenes = {str(self.records[i]['scene_id']) for i in self.train_indices}
        heldout_scenes = {str(self.records[i]['scene_id']) for i in self.heldout_indices}
        if train_scenes & heldout_scenes:
            raise ValueError("Scene split leakage between motion and interaction corpora: "+str(train_scenes & heldout_scenes))

    def __len__(self):
        return len(self.records)

    def batch(self, indices, *, keypose_counts=None, device=None):
        if len(indices)!=1:
            raise ValueError("Use gradient accumulation, not mismatched-anatomy batching")
        source,index=self.lookup[int(indices[0])]
        return self.datasets[source].batch([index],keypose_counts=keypose_counts,device=device)

    def choose(self,rng,interaction_probability=.2):
        if not 0<interaction_probability<=1:
            raise ValueError("Interaction sampling probability must be positive")
        pool=self.train_indices
        if self.motion_train_indices:
            pool=(self.interaction_train_indices if rng.random()<interaction_probability else self.motion_train_indices)
        return int(rng.choice(pool))
```
