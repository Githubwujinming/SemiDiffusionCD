# coding=utf-8
import glob
import os, random, shutil
import os.path as osp 
def check_dir(dir):
    '''检查文件夹是否存在，并创建不存在的文件夹'''
    if not os.path.exists(dir):
        os.mkdir(dir)
# random.seed(0)
# 将图片拆分成训练集train(0.8)和验证集val(0.2)

def moveFile(Dir,splited_dir,train_ratio=0.8,val_ratio=0.1, test_ratio=0.1):

    if not os.path.exists(os.path.join(splited_dir, 'train')):
        os.makedirs(os.path.join(splited_dir, 'train'))
    
    if not os.path.exists(os.path.join(splited_dir, 'val')):
        os.makedirs(os.path.join(splited_dir, 'val'))
    if not os.path.exists(os.path.join(splited_dir, 'test')):
        os.makedirs(os.path.join(splited_dir, 'test'))

    filenames = []
    Adir = os.path.join(Dir,'A')
    Bdir = os.path.join(Dir,'B')
    Ldir = os.path.join(Dir,'label')
    for root,dirs,files in os.walk(Adir):
        for name in files:
            filenames.append(name)
        break
    
    filenum = len(filenames)
    # print(filenames)
    num_train = int(filenum * train_ratio)
    num_val = int(filenum * val_ratio)
    sample_train = random.sample(filenames, num_train)

    for name in sample_train:
        check_dir(os.path.join(splited_dir, 'train','A'))
        check_dir(os.path.join(splited_dir, 'train','B'))
        check_dir(os.path.join(splited_dir, 'train','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'train','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'train','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'train','label', name.replace('A_','L_')))

    sample_val_test = list(set(filenames).difference(set(sample_train)))
    sample_val = random.sample(sample_val_test, num_val)

    for name in sample_val:
        check_dir(os.path.join(splited_dir, 'val','A'))
        check_dir(os.path.join(splited_dir, 'val','B'))
        check_dir(os.path.join(splited_dir, 'val','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'val','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'val','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'val','label', name.replace('A_','L_')))
    sample_test = list(set(sample_val_test).difference(set(sample_val)))

    for name in sample_test:

        check_dir(os.path.join(splited_dir, 'test','A'))
        check_dir(os.path.join(splited_dir, 'test','B'))
        check_dir(os.path.join(splited_dir, 'test','label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'test','A',name))
        shutil.copy(os.path.join(Bdir, name.replace('A_','B_')), os.path.join(splited_dir, 'test','B', name.replace('A_','L_')))
        shutil.copy(os.path.join(Ldir, name.replace('A_','L_')), os.path.join(splited_dir, 'test','label', name.replace('A_','L_')))

    # for name in sample_val:
    #     shutil.move(os.path.join(Dir, name), os.path.join(Dir, 'val'))

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def splitData(Dir,splited_dir,train_ratio=10):
    if not os.path.exists(os.path.join(splited_dir, 'train%s'%train_ratio)):
        os.makedirs(os.path.join(splited_dir, 'train%s'%train_ratio))

    filenames = []
    Adir = os.path.join(Dir,'A')
    Bdir = os.path.join(Dir,'B')
    Ldir = os.path.join(Dir,'label')
    for root,dirs,files in os.walk(Adir):
        for name in files:
            filenames.append(name) 
        break
    filenum = len(filenames)
    # print(filenames)
    num_train = int(filenum * train_ratio/100)
    sample_train = random.sample(filenames, num_train)
    for name in sample_train:
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'A'))
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'B'))
        check_dir(os.path.join(splited_dir, 'train%s'%train_ratio,'label'))
        shutil.copy(os.path.join(Adir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'A',name))
        shutil.copy(os.path.join(Bdir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'B', name))
        shutil.copy(os.path.join(Ldir, name), os.path.join(splited_dir, 'train%s'%train_ratio,'label', name))
        
    # sample_unsup = list(set(filenames).difference(set(sample_train)))
    # write_rel_paths(sample_train)
    # write_rel_paths(sample_unsup)
       

def write_rel_paths(phase, names, out_dir, prefix=''):
    """将文件相对路径存储在txt格式文件中"""
    with open(osp.join(out_dir, phase+'.txt'), 'w') as f:
        for name in names:
            if is_image_file(name):
                f.write(
                    ' '.join([
                        osp.join(name)
                    ])
                )
                f.write('\n')
def gen_datalist(dir, save_name, out_dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    names = os.listdir(osp.join(dir,'label'))
    write_rel_paths(save_name, names, out_dir)

def get_unsupervised_list(all_dir,part_dir, save_name, out_dir):
    all_label = os.listdir(osp.join(all_dir,'label'))
    sup_label = os.listdir(osp.join(part_dir,'label'))
    unsup_label = list(set(all_label).difference(set(sup_label)))
    write_rel_paths(save_name, unsup_label, out_dir)
    
    
def add_prefix(dir, phase, prefix):
    filenames = os.listdir(osp.join(dir, phase,'label'))
    for t in ['A', 'B', 'label']:
        for name in filenames:
            src_path = os.path.join(dir, phase,t,name)
            dst_path = os.path.join(dir, phase,t,prefix + '_' + name)
            os.rename(src=src_path,dst=dst_path)
            # print(src_path, dst_path)

if __name__ == '__main__':
    # Dir = '../datasets/BCD/BCDD_cropped256/'
    # splited_dir = '../datasets/BCD/BCDD_splited_cropped256/'
    # check_dir(splited_dir)
    # moveFile(Dir,splited_dir)
    # for root,dirs,files in os.walk(Dir):
    #     for name in dirs:
    #         folder = os.path.join(root, name)
    #         print("正在处理:" + folder)
    #         # moveFile(folder)
    #     print("处理完成")
    #     break
    all_dir = '/data/datasets/Seasonvarying/train'
    dir = '/data/datasets/Seasonvarying/train50'
    part_dir = '/data/datasets/Seasonvarying/train10'
    save_name = '10_train_unsupervised'
    out_dir='/data/datasets/Seasonvarying/list'
    # gen_datalist(dir, save_name, out_dir)
    get_unsupervised_list(all_dir,part_dir, save_name, out_dir)
    
    # dir = '/data/datasets/Seasonvarying/'
    # add_prefix(dir, 'train50', prefix='train')