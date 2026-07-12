FEATURE_REGISTRY = []

def register_feature(feature_class):
    FEATURE_REGISTRY.append(feature_class)
    return feature_class
