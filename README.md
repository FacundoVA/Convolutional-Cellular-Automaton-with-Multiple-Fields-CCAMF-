# Convolutional-Cellular-Automaton-with-Multiple-Fields-CCAMF-
Convolutional Cellular Automaton with Multiple Fields (CCAMF) is a project that combines cellular automata with convolutional kernels across multiple interacting fields. It simulates complex behaviors like waves, collisions, and gravity in a flexible 2D sandbox, offering real-time interaction and customization for emergent phenomena.




Convolutional Cellular Automaton with Multiple Fields (CCAMF)
Welcome to the Convolutional Cellular Automaton with Multiple Fields (CCAMF)—a project that integrates convolutional techniques with cellular automata across multiple interacting fields. This innovative approach aims to simulate complex behaviors such as wave interactions, collisions, and gravitational effects within a flexible 2D sandbox environment.

Traditional cellular automata often face limitations in representing intricate dynamics due to their simplicity. By incorporating multiple fields and convolutional kernels, CCAMF provides a more general and efficient framework for exploring a wide array of emergent phenomena. This project serves as a platform for experimentation, allowing users to customize interactions and observe how simple rules can lead to complex system behaviors.

Table of Contents
Features
Installation
Usage
Controls
Saving and Loading States
Contributing
License
Acknowledgements
Features
Multiple Interacting Fields: Simulate complex phenomena using multiple fields that interact through convolutional kernels.
Customizable Activation Functions: Choose from activation functions like Tanh and ReLU, and adjust parameters to influence field behaviors.
Dynamic Interaction Kernels: Define how fields influence each other with customizable convolution kernels that can be symmetric or asymmetric.
Interactive Brush Tool: Modify fields in real-time using an adjustable brush size and value to paint directly on the simulation grid.
Real-Time Visualization: Observe the simulation as it evolves, with options to view individual fields or a combined state.
Save and Load States: Save the current state of the simulation to disk and load it later to continue exploring.
Anti-Epilepsy Mode: Reduce rapid flashing by toggling frame skipping, enhancing comfort during extended sessions.
Performance Optimization: Leverage parallel processing with Joblib to handle complex computations efficiently.
User-Friendly Interface: Navigate and control the simulation using intuitive buttons and keyboard shortcuts.
Installation
Prerequisites
Python 3.7 or higher: Download Python
Git: Download Git
Clone the Repository
bash
Copiar código
git clone https://github.com/yourusername/CCAMF.git
cd CCAMF
Create a Virtual Environment (Optional but Recommended)
bash
Copiar código
python -m venv venv
Activate the virtual environment:

Windows:

bash
Copiar código
venv\Scripts\activate
macOS/Linux:

bash
Copiar código
source venv/bin/activate
Install Dependencies
Ensure you have pip updated:

bash
Copiar código
pip install --upgrade pip
Install the required packages:

bash
Copiar código
pip install -r requirements.txt
If the requirements.txt file is not present, install the dependencies manually:

bash
Copiar código
pip install pygame numpy scipy joblib
Usage
Run the simulation using the following command:

bash
Copiar código
python CCAMF.py
Upon launching, a window will appear displaying the simulation grid alongside a control panel with various interactive buttons.

Controls
Mouse Controls:
Left Click: Paint on the grid with the current brush value.
Right Click: Paint on the grid with the negative of the current brush value.
Keyboard Controls:
Spacebar: Toggle Play/Pause the simulation.
Right Arrow: Advance the simulation by one iteration (only when paused).
R: Reset the simulation to its initial state with predefined patterns.
F: Cycle through the available fields to select which field to influence with the brush.
K: Toggle the symmetry of interaction kernels between symmetric and asymmetric.
D: Cycle through display modes (e.g., combined view or individual field views).
S: Save the current simulation state to save_state.pkl.
L: Load a simulation state from save_state.pkl.
E: Toggle Anti-Epilepsy mode to reduce rapid frame updates.
Up/Down Arrow: Increase or decrease the brush size.
Plus/Minus Keys: Increase or decrease the brush value.
Control Panel Buttons:
Brush Size (-/+): Adjust the size of the brush used for painting on the grid.
Brush Value (-/+): Modify the intensity/value applied by the brush.
Field Selection: Display and change the currently selected field for brush interactions.
Kernel Symmetry: Toggle the symmetry of interaction kernels between symmetric and asymmetric.
Save/Load: Save the current state or load a previously saved state.
Display Mode: Switch between combined visualization or viewing individual fields.
Anti-Epilepsy Mode: Toggle frame skipping to minimize rapid flashing.
Visualization Modes
Combined View: Displays the aggregated state of all fields, providing a holistic view of the simulation.
Individual Field Views: Focus on specific fields to observe their unique dynamics and interactions.
Saving and Loading States
Save State
To save the current state of the simulation:

Click the Save button in the control panel or press the S key.
The state will be saved to a file named save_state.pkl in the project directory.
Load State
To load a previously saved simulation state:

Click the Load button in the control panel or press the L key.
Ensure that the save_state.pkl file exists in the project directory.
The simulation will load the state from the file, restoring all fields, activation functions, and interaction kernels.
Note: The loaded state must be compatible with the current simulation configuration (e.g., grid size, number of fields). If discrepancies are detected, an error will be logged, and the load operation will be aborted.

Contributing
Contributions are welcome! Whether you're fixing bugs, improving documentation, or adding new features, your efforts are appreciated.

Fork the Repository

Create a New Branch

bash
Copiar código
git checkout -b feature/YourFeatureName
Commit Your Changes

bash
Copiar código
git commit -m "Add a descriptive commit message"
Push to Your Fork

bash
Copiar código
git push origin feature/YourFeatureName
Open a Pull Request

Provide a clear description of your changes and the motivation behind them.

License
This project is licensed under the MIT License.

Acknowledgements
Pygame: For visualization and user interaction.
NumPy: For efficient numerical computations.
SciPy: For convolution operations.
Joblib: For parallel processing.
Python: The programming language used to develop this simulation.
