from sentence_transformers import SentenceTransformer
import numpy as np

def build_description_embeddings(
    train_text,
    test_text=None,
    model_name: str = "paraphrase-MiniLM-L3-v2",
    batch_size: int = 64,
):
    """
    Build universal description embeddings using a pretrained Sentence-BERT model.

    Args:
        train_text: pandas Series or list of strings (training descriptions)
        test_text:  pandas Series or list of strings (test descriptions), or None
        model_name: name of the SBERT model to use
        batch_size: encoding batch size

    Returns:
        emb_train: np.ndarray of shape (N_train, D)
        emb_test:  np.ndarray of shape (N_test, D) or None
        model:     loaded SentenceTransformer model
    """
    # 1. Load pretrained SBERT
    model = SentenceTransformer(model_name)

    # 2. Encode train descriptions
    train_text_list = [str(t) for t in train_text]
    emb_train = model.encode(
        train_text_list,
        batch_size=batch_size,
        show_progress_bar=True
    ).astype(np.float32)

    # 3. Encode test descriptions with the same model
    emb_test = None
    if test_text is not None:
        test_text_list = [str(t) for t in test_text]
        emb_test = model.encode(
            test_text_list,
            batch_size=batch_size,
            show_progress_bar=True
        ).astype(np.float32)

    return emb_train, emb_test, model
