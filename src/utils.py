import os

def make_dir(input_path):
    try:
        if not os.path.isdir(input_path):
            os.makedirs(os.path.join(input_path))

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def remove_dir(input_path):
    try:
        if os.path.isdir(input_path):
            shutil.rmtree(input_path)

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to delete directory")
            raise

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn
