from train import *


if __name__ == "__main__":
    model_config_mf = {
        "model_mark": "gmf(mf_dim=8)",
        'embedding_dim_mf': 8,
        # "mlp_layers(X)" : 0,
        # 'mlp_layers': [ 32, 16, 8],
        'model_type': 'GMF'     #　MLP, NeuMF
    }

    model_config_mlp = {
        "model_mark": "mlp(mlp_layer=3)",
        "mlp_layers(X)" : 3,
        'mlp_layers': [64, 32, 16, 8],
        'model_type': 'MLP'     #　MLP, NeuMF
    }

    model_config_neumf = {
        "model_mark": "neumf(mf_dim=8,mlp_layer=3)",
        'embedding_dim_mf': 8,
        "mlp_layers(X)" : 3,
        'mlp_layers': [64, 32, 16, 8],
        'model_type': 'NeuMF'     #　MLP, NeuMF
    }

    model_train(model_config_mf, seed = 42, num_of_negatives=4, num_of_epochs = 40)
    print()
    print("-"*30)
    model_train(model_config_mlp, seed = 42, num_of_negatives=4, num_of_epochs = 40)
    print()
    print("-"*30)
    model_train(model_config_neumf, seed = 42, num_of_negatives=4, num_of_epochs = 40)

    # model_config_mlp4 = {
    #     "model_mark": "mlp(mlp_layer=4)",
    #     "mlp_layers(X)" : 4,
    #     'mlp_layers': [128,64, 32, 16, 8],
    #     'model_type': 'MLP'     #　MLP, NeuMF
    # }
    # model_train(model_config_mlp4, seed = 42, num_of_negatives=4, num_of_epochs = 40)
