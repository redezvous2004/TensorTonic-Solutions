import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    _, h_d, w_d, _ = decoder_features.shape
    _, h_e, w_e, _ = encoder_features.shape
    start_h = (h_e - h_d) // 2
    start_w = (w_e - w_d) // 2
    cropped_features = encoder_features[:, start_h: start_h + h_d, start_w: start_w + w_d, :]
    concat_features = np.concatenate((cropped_features, decoder_features), axis=-1)
    return concat_features
