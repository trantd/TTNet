# TTNet a novel machine learning model for facial emotion detection using in online learning systems
TTNet is a lightweight neural network library implemented in Python. It provides a simple and intuitive interface for building and training neural networks. With TTNet, you can quickly prototype and experiment with different network architectures and training configurations.

## Features

- Simple and intuitive API for defining neural networks
- Support for popular network layers such as fully connected, convolutional, and recurrent layers
- Multiple activation functions including ReLU, sigmoid, and tanh
- Common loss functions such as mean squared error and cross-entropy
- Flexible customization options for network architecture and training parameters
- Efficient computation through optimized vectorized operations
- Compatibility with popular deep learning frameworks and libraries

## Installation

To use TTNet, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/chituanh/TTNet.git

2. Install the required dependencies:
pip install -r requirements.txt

3. Import TTNet into your Python project:
import ttnet
## Usage
To create a neural network using TTNet, follow these steps:
1. Define the architecture of your network by stacking layers:
model = ttnet.Sequential()
model.add(ttnet.Linear(784, 256))
model.add(ttnet.ReLU())
model.add(ttnet.Linear(256, 10))
model.add(ttnet.Softmax())
2. Configure the training process:
model.compile(loss=ttnet.CrossEntropyLoss(), optimizer=ttnet.SGD(lr=0.01))
3. Train the network on your data:
model.train(X_train, y_train, epochs=10, batch_size=32)
4. Evaluate the performance of the trained model:
accuracy = model.evaluate(X_test, y_test)
For more detailed instructions and API documentation, refer to the official documentation

## Examples
Check out the examples directory in the repository for various usage examples and demonstrations of TTNet on different tasks.

## Contributing
Contributions to TTNet are welcome! If you find any issues or have ideas for improvements, please open an issue or submit a pull request. See the contribution guidelines for more information.

## License
TTNet is licensed under the MIT License. See the LICENSE file for more details.
Feel free to use this template as a starting point for your software's README.md file. Customize and expand upon it according to your specific project requirements. Good luck with your software development!

## Dataset for test the software
 - Data Fer-2013: [Fer-2013](https://www.kaggle.com/datasets/msambare/fer2013)# TTNet
