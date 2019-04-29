from model_k import *

def menu():
    menu=int(input("""
    0. Split dataset between training, evaluation and prediction randomly.
    1. Define model first Time.
    2. Load model.
    3. Train and evaluate.
    4. Predict.
    5. Exit.

Select an option: """))
    return menu

while True:
    m=menu()
    if m==5:
        break

    if m==0:
        dataset_dir=input("Dataset Dir[dataset]: ") or "dataset"
        train_percent=float(input("Train percent[0.7]: ") or 0.7)
        evaluate_percent=float(input("Evaluate percent[0.2]: ") or 0.2)
        seed=input("Seed[None]: ") or None
        if seed!=None:
            seed=int(seed)
        train_evaluate_predict_split(dataset_dir, train_percent, evaluate_percent, seed)
    if m==1:
        def_model()
        print("Model defined: optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']")
    if m==2:
        model_name=input("Model Name to load[model_curves]: ") or "model_curves"
        load_model(model_name)
    if m==3:
        dataset_dir_train=input("Dataset Dir[train]: ") or "train"
        epoch=int(input("Epoch[100]: ") or 100)
        batch_size=int(input("Batch size[2000]: ") or 2000)
        model_name=input("Model Name[model_curves]: ") or "model_curves"
        train_and_evaluate(dataset_dir_train, epoch, batch_size, model_name)
    if m==4:
        predict()
