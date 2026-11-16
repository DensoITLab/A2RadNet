"""Build the proposed model while preserving historical state-dict names."""
import inspect


def build_model(config):
    dataset = config['dataset_name']
    if config['data_mode'] != 'RA':
        raise ValueError('The proposed method consumes complex RD tensors (legacy data_mode=RA).')
    if dataset == 'radial':
        from .radial import A2RadNet
    elif dataset == 'raddet':
        from .raddet import A2RadNet
    else:
        raise ValueError('dataset_name must be radial or raddet')
    model = config['model']
    expected = 'doppler_structured' if dataset == 'radial' else 'regular_antenna'
    if model.get('aae_variant', expected) != expected:
        raise ValueError('AAE variant does not match the dataset configuration.')
    if bool(model.get('frontend_use_cyclic_shift', True)) != (dataset == 'radial'):
        raise ValueError('Use circular projection for RADIal and pointwise projection for RADDet.')
    if model.get('frontend_type') != 'MIMO_MLP':
        raise ValueError('The release supports the learned AAE frontend only.')
    if model.get('frontend_angle_estimator_type', 1) != 1:
        raise ValueError('Only the paper AAE projection is included.')
    if model.get('frontend_doppler_reducer', 0) not in (0, 1):
        raise ValueError('Only the paper DLR or no DLR is included.')
    if any(model.get('use_periodic_attention', [])):
        raise ValueError('The paper uses spatial Swin attention, not periodic attention.')
    if not model['DetectionHead'] or bool(model['SegmentationHead']) != (dataset == 'radial'):
        raise ValueError('Use detection + freespace on RADIal, detection + classification on RADDet.')
    if len(config['losses']['weight']) < 3 or any(config['losses']['weight'][3:]):
        raise ValueError('Expected three weights: detection, regression, and the dataset-specific task.')
    parameters = inspect.signature(A2RadNet).parameters
    kwargs = {key: value for key, value in model.items() if key in parameters}
    kwargs.update(detection_head=model['DetectionHead'], segmentation_head=model['SegmentationHead'])
    if dataset == 'radial':
        kwargs['label_map'] = config['label_map']
    return A2RadNet(**kwargs)
