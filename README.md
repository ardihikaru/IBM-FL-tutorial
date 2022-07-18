# IBM-FL-tutorial

This repository simplifies the user guide given on the official IBM-FL repository

## setup

* install `anaconda`
* Create conda env
  ```shell
  conda create -n fltest1 python=3.6 tensorflow==1.15.0
  ```
* Activate conda env
  ```shell
  conda activate fltest1
  ```
* Deactivate conda env
  ```shell
  conda deactivate
  ```
* Install the IBM FL package (`*.whl` file is in the `/federated-learning-lib` folder)
    * WHL file name: `federated_learning_lib-1.0.7-py3-none-any.whl`
    * **IMPORTANT**: This WHL file **DOES NOT COMPATIBLE** with `tensorflow=2.1.0`
    * You can download it [officially here](https://github.com/IBM/federated-learning-lib/tree/main/federated-learning-lib) 
      or [here](https://drive.google.com/file/d/1WKPU22hz5J6eD8Fzvknr-OZ6qmm5woQJ/view?usp=sharing)
    * Put the `*.whl` file in `./IBM-FL-tutorial/whl_files` folder
    * Install the WHL file:
      ```shell
      pip install whl_files/federated_learning_lib-1.0.7-py3-none-any.whl
      ```

## Usage step-by-step

In these steps, I follow [this tutorial](https://github.com/IBM/federated-learning-lib/blob/main/setup.md)

* Split Sample Data
    * **generate sample data on any of the integrated datasets**
      ```shell
      python examples/generate_data.py -n 2 -d mnist -pp 200
      ```
        * Resulted with following logs
          ```shell
          /home/ardi/anaconda3/envs/fl-py36-ts1150/lib/python3.6/site-packages/urllib3/connectionpool.py:1052: 
          InsecureRequestWarning: Unverified HTTPS request is being made to host 's3.amazonaws.com'. 
          Adding certificate verification is strongly advised. 
          See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings
          InsecureRequestWarning, 
          Warning: test set and train set contain different labels
          Party_ 0
          nb_x_train:  (200, 28, 28) nb_x_test:  (5000, 28, 28)
          * Label  0  samples:  26
          * Label  1  samples:  26
          * Label  2  samples:  17
          * Label  3  samples:  24
          * Label  4  samples:  15
          * Label  5  samples:  20
          * Label  6  samples:  19
          * Label  7  samples:  17
          * Label  8  samples:  19
          * Label  9  samples:  17
          Finished! :) Data saved in  examples/data/mnist/random
          Party_ 1
          nb_x_train:  (200, 28, 28) nb_x_test:  (5000, 28, 28)
          * Label  0  samples:  17
          * Label  1  samples:  22
          * Label  3  samples:  26
          * Label  4  samples:  21
          * Label  2  samples:  18
          * Label  5  samples:  18
          * Label  6  samples:  21
          * Label  7  samples:  22
          * Label  8  samples:  19
          * Label  9  samples:  16
          Finished! :) Data saved in  examples/data/mnist/random
          ```
        * This command would generate 2 parties with 200 data points each from the MNIST dataset
            * File 1: `examples/data/mnist/random/data_party0.npz`
            * File 2: `examples/data/mnist/random/data_party1.npz`
        * File tree will be like this:
          ```shell
          ./IBM-FL-tutorial/examples/data
          ...
          │
          └─── mnist
              │
              └─── random
                   │ data_party0.npz
                   │ data_party1.npz
          ...
          ```
    * Create Configuration Files
        * You can generate these config files using the `generate_configs.py` script.
        * Run command:
          ```shell
          python examples/generate_configs.py -f iter_avg -m keras -n 2 -d mnist -p examples/data/mnist/random
          ```
            * List of models: `{keras,pytorch,tf,sklearn,doc2vec,None}`
            * Tried with `tf` but got error
              ```shell
              OMP: Info #172: KMP_AFFINITY: OS proc 14 maps to socket 0 core 7 thread 0 
              OMP: Info #172: KMP_AFFINITY: OS proc 15 maps to socket 0 core 7 thread 1 
              OMP: Info #255: KMP_AFFINITY: pid 212690 tid 212690 thread 0 bound to OS proc set 0
              Finished generating config file for aggregator. Files can be found in:  /home/ardi/devel/nycu/fl/federated-learning-lib/examples/configs/iter_avg/tf/config_agg.yml
              WARNING:tensorflow:From /home/ardi/anaconda3/envs/fl-py36-ts1150/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
              Instructions for updating:
              If using Keras pass *_constraint arguments to layers.
              Traceback (most recent call last):
                File "examples/generate_configs.py", line 297, in <module>
                  dataset, party_data_path, folder_configs, task_name)
                File "examples/generate_configs.py", line 228, in generate_party_config
                  'model': generate_model_config(module, model, folder_configs, dataset, party_id=i),
                File "examples/generate_configs.py", line 170, in generate_model_config
                  model = get_model_config(folder_configs, dataset, is_agg, party_id, model=model)
                File "/home/ardi/devel/nycu/fl/federated-learning-lib/examples/iter_avg/generate_configs.py", line 69, in get_model_config
                  return method(folder_configs, dataset, is_agg=is_agg, party_id=0)
                File "/home/ardi/devel/nycu/fl/federated-learning-lib/examples/iter_avg/model_tf.py", line 44, in get_model_config
                  model.compute_output_shape(input_shape=input_shape)
                File "/home/ardi/anaconda3/envs/fl-py36-ts1150/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/network.py", line 699, in compute_output_shape
                  return super(Network, self).compute_output_shape(input_shape)
                File "/home/ardi/anaconda3/envs/fl-py36-ts1150/lib/python3.6/site-packages/tensorflow_core/python/keras/engine/base_layer.py", line 646, in compute_output_shape
                  raise NotImplementedError
              NotImplementedError
              ```
        * Resulted with following logs:
          ```shell
          ...
          Finished generating config file for parties. Files can be found in:  
          /home/ardi/devel/nycu/fl/federated-learning-lib/examples/configs/iter_avg/keras/config_party*.yml
          ```
        * File tree will be like this:
          ```shell
          ./IBM-FL-tutorial/examples/configs
          ...
          │
          └─── iter_avg
              │
              └─── keras
                   │ compiled_keras.h5
                   │ config_agg.yml
                   │ config_party0.yml
                   │ config_party1.yml
          ...
          ```
    * Initiate Learning
        * **Start the Aggregator**
            * To start the aggregator, open a terminal window running the IBM FL environment set up previously.
                1. In the terminal run:
                   ```shell
                    python -m ibmfl.aggregator.aggregator examples/configs/iter_avg/keras/config_agg.yml
                   ```
                    * where the path provided is the aggregator config file path.
                2. Then in the terminal, type `START` and press enter.
        * **Register Parties**
            * To register new parties, open a new terminal window for each party,
              running the IBM FL environment set up previously.
                1. In the terminal run:
                   ```shell
                   python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party0.yml
                   ```
                    * where the path provided is the path to the party config file.
                    * **NOTE**: Each party will have a different config file,
                      usually noted by changing `config_party<idx>.yml`
                    * Since I have two parties, I run another party:
                      ```shell
                      python -m ibmfl.party.party examples/configs/iter_avg/keras/config_party1.yml
                      ```
                2. Then in the terminal, type `START` and press enter.
                    * Do the same thing to the other party. :)
                3. Then in then terminal for each party, type `REGISTER` and press enter.
                    * Do the same thing to the other party. :)
        * **Perform Data Training**
            * To initiate federated training, type `TRAIN` in your aggregator terminal and press enter.
        * **Sync**
            * To synchronize model among parties, type `SYNC` in aggregator terminal
        * **Save**
            * To save the model, type `SAVE` in your aggregator terminal, and press enter.
            * Response from `party0`
              ```shell
              ...
              2022-07-18 20:41:42,765 | 1.0.6 | INFO | ibmfl.model.keras_fl_model                    
              | Model saved in path: 
              /home/ardi/devel/nycu/fl/federated-learning-lib/keras-cnn_1658148102.4956248.h5.
              ...
              ```
            * Response from `party1`
              ```shell
              ...
              2022-07-18 20:41:42,760 | 1.0.6 | INFO | ibmfl.model.keras_fl_model                    
              | Model saved in path: 
              /home/ardi/devel/nycu/fl/federated-learning-lib/keras-cnn_1658148102.5107014.h5.
              ...
              ```
            * Now models for each party are saved to your folder.
              **In the real scenario, each party would just see one model file for their party.**
            * Now we have the model that is ready to be used for prediction!
            * File tree will be like this:
              ```shell
              ./IBM-FL-tutorial
              ...
              │ keras-cnn_1658148102.4956248.h5
              │ keras-cnn_1658148102.5107014.h5
              ...
              ```
                * **FYI**: The filename `*.h5` will always be different.
        * **Stop**
            * Remember to type `STOP` in both the aggregator terminal and parties’ terminal
              to end the experiment process.
            * Congratulation! Your party has just finished the training process without revealing your data.
        * **Prediction**
            * Imagine you are party 0, to use the model to predict the test data of your party,
            * all you need to do is to load the model and predict the test data of your party.

## Process analysis inside IBM-FL

* When `AGGREGATOR` submit `SAVE`
    * `PARTY-X` received `POST /8`
* When `PARTY-X` submit `REGISTER`
    * POST /6 masuk di AG
* When `AGGREGATOR` submit `TRAIN`
    * `AGGREGATOR` received `POST /7` (**updating model**)
    * `PARTY-X` received `POST /7` (**updating model**)

## MISC

* [MNIST sample data](https://drive.google.com/drive/folders/1S9zHlZLLPEnEfMVsPKgqIqgn14PE8zku?usp=sharing)
  * Put these files inside `./IBM-FL-tutorial/examples/data` folder
* [Keras models](https://drive.google.com/drive/folders/1jfMKRqWdCK6k2sNIPKHdDt-6T1Mjdiap?usp=sharing)
  * Put these files inside `./IBM-FL-tutorial` folder (**ROOT DIRECTORY**)
* [IBM-FL WHL file](https://drive.google.com/file/d/1WKPU22hz5J6eD8Fzvknr-OZ6qmm5woQJ/view?usp=sharing)
  * Or download from the official link [here](https://github.com/IBM/federated-learning-lib/tree/main/federated-learning-lib)
  * Put these files inside `./IBM-FL-tutorial/whl_files` folder
