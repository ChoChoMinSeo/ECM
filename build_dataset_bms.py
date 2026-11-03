import torch.utils
from src.datasets.datasets import *
import torch
from torchvision import transforms

def get_train_dataset(args, subsample = 1, max_len = 100):
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(args.patch_size,pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    datasets = []
    if 'Kodak' in args.tasks:
        if args.wk == 'local':
            path = [
                "./data/CLIC2020/train/mobile_train_2020/train/","./data/CLIC2020/train/professional_train_2020/train/",\
            ]
        datasets.append(HQDataset(path, train_transforms, 'train',subsample, (max_len*args.natu)//2))
        datasets[-1].cls_idx = onehot_convert([0], max_cls = args.n_classes, eta = args.eta)
        if args.wk == 'local':
            path = [
                "./data/Flicker2W/train.list"
            ]
        datasets.append(HQDataset(path, train_transforms, 'train',subsample, (max_len*args.natu)//2))
        datasets[-1].cls_idx = onehot_convert([0], max_cls = args.n_classes, eta = args.eta)
        
    if 'Document' in args.tasks:
        if args.wk == 'local':
            path = "./data/2024NeurIPS/"
        datasets.append(DocumentDataset(path, train_transforms, 'train',subsample, max_len*args.docu))        
        datasets[-1].cls_idx = onehot_convert([1], max_cls = args.n_classes, eta = args.eta)
        
    if 'Residue' in args.tasks:
        if args.wk == 'local':
            path = "./data/vimeo_triplet/"
        datasets.append(VimeoTripletResidueDataset(path, train_transforms, 'train',subsample, max_len*args.diff))     
        datasets[-1].cls_idx = onehot_convert([2], max_cls = args.n_classes, eta = args.eta)
        
    if 'Y1Y2Y3' in args.tasks:
        if args.wk == 'local':
            path = "./data/vimeo_triplet/"
        datasets.append(VimeoTripletY1Y2Y3Dataset(path, train_transforms, 'train',subsample, max_len*args.moti)) 
        datasets[-1].cls_idx = onehot_convert([3], max_cls = args.n_classes, eta = args.eta)
        
    if 'KITTI' in args.tasks:
        if args.wk == 'local':
            path = "./data/Kitti/Kitti/"
        datasets.append(KittiDrivingDataset(path, train_transforms, 'train',subsample, max_len*args.kitt))
        datasets[-1].cls_idx = onehot_convert([4], max_cls = args.n_classes, eta = args.eta)
        
    if 'LineDrawing' in args.tasks:
        if args.wk == 'local':
            path = "./data/Manga109/Manga109_released_2023_12_07/"
        datasets.append(LineDataset(path, train_transforms, 'train',subsample, max_len*args.line))
        datasets[-1].cls_idx = onehot_convert([5], max_cls = args.n_classes, eta = args.eta)
        
    dataset = torch.utils.data.ConcatDataset(datasets)
    return CutMix(dataset, num_class=args.n_classes, prob=args.cutmix_prob)

def get_test_dataset(args):
    test_transforms2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    datasets = []
    names = []       
    if 'Cutmix' in args.test_tasks:
        if args.wk =='local':
            paths = {
                0: "./data/test/00_Kodak/",
                1: "./data/test/01_Document/",
                2: "./data/test/02_Residue/",
                3: "./data/test/03_Y1Y2Y3/",
                4: "./data/test/04_KITTI/",
                5: "./data/test/05_LineDrawing/",
            }
        datasets.append(HQDataset_cutmix_test(paths[args.base_m],paths[args.add_m], test_transforms2, 'test'))#0
        datasets[-1].cls_idx= onehot_convert([args.base_m],max_cls=args.n_classes, eta = args.eta)
        datasets[-1].cls_idx_add= onehot_convert([args.add_m],max_cls=args.n_classes, eta = args.eta)
        names.append('Cutmix')
        
    return datasets, names
    
dataset_name = ['Kodak','Document','Residue','Y1Y2Y3', 'KITTI','LineDrawing']
name_to_token_idx = {
    'Kodak':0,
    'Document':1,
    'Residue':2,
    'Y1Y2Y3': 3,
    'KITTI':4,
    'LineDrawing':5,
}
token_idx_to_name = {
    0:'Kodak',
    1:'Document',
    2:'Residue',
    3:'Y1Y2Y3',
    4:'KITTI',
    5:'LineDrawing',
}
task_colors = {
    'Kodak': 'blue', 
    'natural_clic': 'red',   
    'Residue': 'yellow',       
    'Y1Y2Y3': '#808080',   # gray
    'Document': 'green',           
    'KITTI': 'cyan',       
    'LineDrawing':'orange'
}

initial_token_stats = {
    'Kodak': {
        'rate':[0.130501825983326, 0.198247019511958, 0.298263602269192, 0.436732123295466, 0.613576415305336, 0.839287536839644],
        "psnr":[29.639518737793, 31.1419787406921, 32.6875612735748, 34.4089248975118, 36.1523416837056, 37.8984320958455]
    },
    'Y1Y2Y3': {
        'rate':[0.118075927866525, 0.174807683025536, 0.250423573153584, 0.356458780551819, 0.491168009803483, 0.657239772771534],
        "psnr":[30.2256191906176, 31.623153887297, 32.9919894619992, 34.3618637386121, 35.6529975188406, 36.8295918765821]
    },
    'Residue': {
        'rate':[0.100120745697304, 0.150023099623228, 0.212542727025912, 0.298987055883596, 0.408005869898357, 0.544237288940502],
        "psnr":[31.9686923779939, 33.402616049114, 34.8648554400394, 36.4047510749415, 37.8728547849153, 39.3316957072208]
    },
    'Document': {
        'rate':[0.127059356775135, 0.172528396360576, 0.198893185394506, 0.256420342251658, 0.325576970974604, 0.412838665768504],
        "psnr":[32.7577881813049, 34.9365599950155, 37.0336861610413, 39.1689858436584, 41.2791113853455, 43.2500732739767]
    },
    'KITTI': {
        'rate':[0.129022969227088, 0.197425905222956, 0.293146602025157, 0.430773776612784, 0.614889005297109, 0.847076833248138],
        "psnr":[28.0591971748754, 29.1222340935155, 30.1910824524729, 31.2466706225747, 32.2122966866744, 33.0488339474327]
    },
    'LineDrawing': {
        'rate':[0.25854535587132, 0.355178575380705, 0.481705743237399, 0.663065082626417, 0.878920761519112,1.13822427997366],
        "psnr":[28.3602757453918, 30.1024646162987, 31.9450402259827, 33.9956914782524, 36.2073285579681, 38.6068336963654]
    },
    'natural_clic': {
        'rate':[0.088690736353761, 0.131515826121336, 0.189950830507569, 0.271658880681526, 0.379913284102591, 0.5270599167],
        "psnr":[32.1219664783013, 33.5282310392798, 34.9188583187941, 36.3877449966058, 37.8466632191728, 39.33175872]
    }
}

trained_token_stats = {
    "Kodak":{
        'rate': [0.1371796845148007, 0.2080245769272247, 0.3079585768282413, 0.4477187780042489, 0.6286589403947195, 0.85568185398976],
        'psnr': [29.521159251530964, 31.002658049265545, 32.56726908683777, 34.28773474693298, 36.06425635019938, 37.83280038833618]
    },
    "Y1Y2Y3":{
        'rate': [0.1075752908854108, 0.1551489082998351, 0.2196550619739451, 0.3112114061062273, 0.4368954547925999, 0.6075145912013555],
        'psnr': [31.15329529109754, 32.635539406224304, 33.99782539668836, 35.44716463590923, 36.835951679631286, 38.14159784818951]
    },
    'Residue':{
        'rate': [0.0953943430274529, 0.1428328264798773, 0.2064025374129414, 0.2921351786016634, 0.4001234906578534, 0.534626066341604],
        'psnr': [32.16470291740016, 33.65236975017347, 35.13258903904965, 36.694830091376055, 38.24163266232139, 39.73907867230867]
    },
    'Document':{
        'rate': [0.0783345303498208, 0.0991960462803642, 0.1202839026227593, 0.1487538544461131, 0.1828772549827893, 0.2297426089644432],
        'psnr': [35.66483926773071, 37.58302434285482, 39.35381778081258, 41.11909008026123, 42.8426251411438, 44.426762421925865]
    },
    'KITTI':{
        'rate': [0.1415770238167361, 0.2116655217189537, 0.3087258456568968, 0.4695578201820976, 0.6588997401689228, 0.905761550131597],
        'psnr': [28.5846629895662, 29.949489944859558, 31.26581146842555, 32.6512716694882, 33.89662787788793, 35.2559535377904]
    },
    'LineDrawing':{
        'rate': [0.2339989455649629, 0.313197270152159, 0.4282571810763329, 0.601058260188438, 0.8218532761093229, 1.0627217558212578],
        'psnr': [28.940471053123478, 30.54675370454788, 32.275451600551605, 34.32549703121185, 36.625941157341, 39.029648542404175]
    },
    'natural_clic':{
        'rate': [0.0894838274433845, 0.1323823767100892, 0.1914897220345532, 0.2732043711514008, 0.3838609851351598, 0.5316923604142375],
        'psnr': [32.0630507120272, 33.47868951937048, 34.86869993442443, 36.33884699751691, 37.81858425605588, 39.31100352217511]
    }
}