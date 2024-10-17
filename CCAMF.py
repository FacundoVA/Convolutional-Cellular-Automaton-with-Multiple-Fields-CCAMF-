import pygame
import numpy as np
import sys
import logging
import random
import pickle
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
from typing import List, Tuple, Optional, Union, Callable
import cProfile
import pstats

# Configure logging for better debug information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define global constants for colors and cell size
COLORS = {
    "background": (10, 10, 40),
    "grid": (50, 50, 80),
    "text": (255, 255, 255),
    "button": (100, 100, 150),
    "button_hover": (150, 150, 200),
    "message": (255, 255, 255),
    "error": (255, 0, 0)
}

CELL_SIZE = 3  # Size of each cell in pixels

# Constants for brush control
MAX_BRUSH_SIZE = 80  # Ensure it's odd for centering
BRUSH_SIZE_INCREMENT = 2  # Increment brush size by 2 to keep it odd
BRUSH_VALUE_INCREMENT = 0.1  # Increment for brush value adjustments




def real_random():
    choice = random.choice([1, 2, 3, 4])  # Precompute the choice to avoid repeated calls

    if choice == 1:
        scale = np.float32(random.uniform(-1, 1))
    elif choice == 2:
        scale = np.float32(random.gauss(0, 0.0001))
    elif choice == 3:
        scale = np.float32(random.gauss(0, 0.001))
    else:
        scale = np.float32(random.gauss(0, 0.01))
        
    return scale



def draw_circle(array: np.ndarray, center_x: float, center_y: float, radius: float, value: float) -> None:
    """
    Draws a circle on a 2D NumPy array.

    Parameters:
    - array (np.ndarray): The 2D NumPy array to modify.
    - center_x (float): X-coordinate of the circle's center.
    - center_y (float): Y-coordinate of the circle's center.
    - radius (float): Radius of the circle.
    - value (float): The value to assign to the circle's pixels.
    """
    center_x = int(center_x)
    center_y = int(center_y)
    radius = int(radius)
    Y, X = np.ogrid[:array.shape[0], :array.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    array[mask] = value
    logger.debug(f"Drew circle at ({center_x}, {center_y}) with radius {radius} and value {value}.")

def draw_grid(surface: pygame.Surface, width: int, height: int, cell_size: int) -> None:
    """
    Draws grid lines on the Pygame surface.

    Parameters:
    - surface (pygame.Surface): The surface to draw the grid on.
    - width (int): Number of cells horizontally.
    - height (int): Number of cells vertically.
    - cell_size (int): Size of each cell in pixels.
    """
    for x in range(0, width * cell_size, cell_size):
        pygame.draw.line(surface, COLORS["grid"], (x, 0), (x, height * cell_size))
    for y in range(0, height * cell_size, cell_size):
        pygame.draw.line(surface, COLORS["grid"], (0, y), (width * cell_size, y))
    logger.debug("Grid lines drawn on the surface.")

class Field:
    """
    Represents a single field in the simulation.
    """

    # Activation functions as class methods
    def tanh_activation(self, x: np.ndarray) -> np.ndarray:
        a, b = self.activation_params
        return a * np.tanh(b * x)

    def relu_activation(self, x: np.ndarray) -> np.ndarray:
        a, b = self.activation_params
        return a * np.maximum(0, b * x)

    # Registry of available activation functions
    ACTIVATION_FUNCTIONS = {
        'tanh': 'tanh_activation',
        'relu': 'relu_activation',
        # Additional activation functions can be added here
    }

    def __init__(
        self,
        width: int,
        height: int,
        max_value: float = 10.0,
        field_index: int = 0,
        symmetric_kernels: bool = True,
        initial_patterns: Optional[List[dict]] = None,
    ):
        """
        Initializes the Field.

        Parameters:
        - width (int): Width of the grid.
        - height (int): Height of the grid.
        - max_value (float): Maximum value for cell states.
        - field_index (int): Index of the field.
        - symmetric_kernels (bool): Whether kernels are symmetric.
        - initial_patterns (List[dict], optional): List of dictionaries specifying initial patterns.
        """
        self.width = width
        self.height = height
        self.max_value = max_value
        self.field_index = field_index
        self.symmetric_kernels = symmetric_kernels

        # Initialize the state of the field with zeros
        self.state = np.zeros((self.height, self.width), dtype=np.float32)
        logger.debug(f"Initialized field {self.field_index} with shape ({self.height}, {self.width}).")

        # Apply initial patterns if provided
        if initial_patterns:
            for pattern in initial_patterns:
                self._apply_initial_pattern(**pattern)
        else:
            # Default initial patterns
            self._apply_default_initial_patterns()

        # Activation function parameters
        self.activation_params: np.ndarray = self.initialize_activation_params()

        # Activation function name
        self.activation_function_name: str = 'tanh'  # Default activation function name

        # Choose an activation function
        self.set_activation_function(self.activation_function_name)
        logger.debug(f"Field {self.field_index} initialized with activation function '{self.activation_function_name}'.")

    def _apply_initial_pattern(self, center: Tuple[float, float], radius: float, value: float) -> None:
        """
        Applies an initial pattern to the field's state.
        """
        center_x, center_y = center
        draw_circle(self.state, center_x, center_y, radius, value)
        logger.info(f"Applied initial pattern at ({center_x}, {center_y}) with radius {radius} and value {value}.")

    def _apply_default_initial_patterns(self) -> None:
        """
        Applies default initial patterns to the field's state.
        """
        # Define default patterns relative to grid size
        patterns = [
            {
                'center': (0.7 * self.width, 0.7 * self.height),
                'radius': 0.02 * min(self.width, self.height),
                'value': 0.666 * self.max_value / 3,
            },
            {
                'center': (0.25 * self.width, 0.3 * self.height),
                'radius': 0.01 * min(self.width, self.height),
                'value': -self.max_value / 3,
            },
        ]

        for pattern in patterns:
            self._apply_initial_pattern(**pattern)

    def initialize_activation_params(self) -> np.ndarray:
        """
        Initializes activation function parameters 'a' and 'b'.
        """
        a = real_random() * self.max_value*1000
        b = real_random() * self.max_value*1000
        print("parametros")
        print(a)
        print(b)
        logger.debug(f"Initialized activation params for field {self.field_index}: a={a}, b={b}.")
        return np.array([a, b], dtype=np.float32)

    def update(self, influence: np.ndarray) -> np.ndarray:
        """
        Updates the state of the field based on the aggregated influence from other fields.
        """
        if influence.shape != self.state.shape:
            raise ValueError(
                f"Influence shape {influence.shape} does not match field state shape {self.state.shape}."
            )

        # Apply activation function
        activated_state = self.activation_function(influence)
        logger.debug(f"Applied activation function for field {self.field_index}.")

        # Clip to max_value
        clipped_state = np.clip(activated_state, -self.max_value, self.max_value)

        # Round to 3 decimal places to limit precision
        new_state = np.round(clipped_state, decimals=3)
        logger.debug(f"Updated state for field {self.field_index} after clipping and rounding.")

        return new_state

    def set_activation_function(self, func_name: str) -> None:
        """
        Sets the activation function to be used.
        """
        if func_name.lower() in self.ACTIVATION_FUNCTIONS:
            # Bind the selected activation function
            self.activation_function_name = func_name.lower()
            method_name = self.ACTIVATION_FUNCTIONS[func_name.lower()]
            self.activation_function = getattr(self, method_name)
            logger.info(f"Set activation function to '{func_name}' for field {self.field_index}.")
        else:
            raise ValueError(
                f"Unsupported activation function '{func_name}'. Available functions: {list(self.ACTIVATION_FUNCTIONS.keys())}"
            )

    def set_activation_params(self, a: float, b: float) -> None:
        """
        Sets the activation function parameters.
        """
        self.activation_params = np.array([a, b], dtype=np.float32)
        logger.info(f"Set activation parameters for field {self.field_index}: a={a}, b={b}.")

    def reset_state(self, randomize: bool = True, retain_patterns: bool = False) -> None:
        """
        Resets the field state.
        """
        if randomize:
            self.state = np.random.uniform(-self.max_value, self.max_value, (self.height, self.width)).astype(np.float32)
            logger.info(f"Field {self.field_index} state reset to random values.")
        else:
            self.state = np.zeros((self.height, self.width), dtype=np.float32)
            logger.info(f"Field {self.field_index} state reset to zeros.")

        if retain_patterns:
            self._apply_default_initial_patterns()
            logger.info(f"Field {self.field_index} retained initial patterns after reset.")

class Simulation:
    """
    Manages the logic of the cellular automaton with multiple interacting fields.
    """

    def __init__(
        self,
        width: int,
        height: int,
        num_fields: int = 5,
        max_value: float = 10.0,
        symmetric_kernels: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initializes the Simulation.
        """
        if num_fields <= 0:
            raise ValueError("Number of fields must be positive.")

        self.width = width
        self.height = height
        self.num_fields = num_fields
        self.max_value = max_value
        self.symmetric_kernels = symmetric_kernels

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Initialize fields with unique parameters
        self.fields: List[Field] = [
            Field(
                width=width,
                height=height,
                max_value=max_value,
                field_index=i,
                symmetric_kernels=symmetric_kernels,
            )
            for i in range(num_fields)
        ]

        # Initialize interaction kernels
        self.interaction_kernels: List[List[List[np.ndarray]]] = self.initialize_interaction_kernels()

    def initialize_interaction_kernels(self) -> List[List[List[np.ndarray]]]:
        """
        Initializes the interaction kernels that define how each field influences others.
        """
        interaction_kernels = []
        for i in range(self.num_fields):
            row = []
            print("Escalas")
            for j in range(self.num_fields):
                if i == j:
                    # Self-interaction: stronger influence to maintain field's pattern
                    scale = real_random()*10
                    
                else:
                    # Cross-interaction: scaled down influence
                    scale = real_random()  # Adjust scale as needed
                print(scale)
                    
                

                
                kernels = self.generate_interaction_kernels(scale=scale)
                row.append(kernels)  # Keep all kernels without summing
                logger.debug(f"Initialized interaction kernels for field {i} influenced by field {j}.")
            interaction_kernels.append(row)
        logger.debug("Initialized interaction kernels for all field interactions.")
        return interaction_kernels

    def generate_interaction_kernels(self, scale: float = 1.0) -> List[np.ndarray]:
        """
        Generates a list of convolution kernels for field interactions.
        Ensures that each kernel has a maximum of 3 decimal places.
        """
        valid_combinations = [[3], [5], [7], [3, 5], [3, 7], [5, 7]]
        kernel_sizes = random.choice(valid_combinations)
        kernels = []
        for size in kernel_sizes:
            
            if random.choice([True, False]):
                kernel = np.random.uniform(-1, 1, (size, size)).astype(np.float32)
            elif random.choice([True, False]):
                kernel = np.random.normal(0, 0.0001, (size, size)).astype(np.float32)
            elif random.choice([True, False]):
                
                kernel = np.random.normal(0, 0.0000001, (size, size)).astype(np.float32)
            else:
                
                kernel = np.random.normal(0, 0.01, (size, size)).astype(np.float32)
            
            
            if self.symmetric_kernels:
                kernel = (kernel + np.flip(kernel)) / 2
                

            # Normalize the kernel to prevent value explosion or decay
            #kernel_sum = np.sum(np.abs(kernel))
            #if kernel_sum != 0:
            #    kernel /= kernel_sum
            # Apply scaling factor
            kernel *= scale
            # Round kernel values to 6 decimal places
            kernel = np.round(kernel, decimals=6)
            kernels.append(kernel)
        logger.debug(f"Generated {len(kernels)} interaction kernels with scale {scale}.")
        return kernels

    def update(self) -> None:
        """
        Updates all fields based on their own state and the influence of other fields.
        Utilizes parallel processing to speed up computations.
        """
        # Parallelize the field updates
        new_states = Parallel(n_jobs=-1)(
            delayed(self.update_single_field)(i) for i in range(self.num_fields)
        )
        # Assign new states
        for i, new_state in enumerate(new_states):
            self.fields[i].state = new_state
            logger.debug(f"Field {i} state updated.")

    def update_single_field(self, i: int) -> np.ndarray:
        """
        Updates a single field by aggregating influences from all other fields.
        """
        field = self.fields[i]
        total_influence = np.zeros((self.height, self.width), dtype=np.float32)
        for j, other_field in enumerate(self.fields):
            kernels = self.interaction_kernels[i][j]
            for kernel in kernels:
                try:
                    conv_result = fftconvolve(other_field.state, kernel, mode='same')
                    total_influence += conv_result
                except Exception as e:
                    logger.error(f"Error during convolution in field {i} influenced by field {j}: {e}")
                    continue
        # Round the total influence to 3 decimal places to prevent floating-point explosion
        total_influence = np.round(total_influence, decimals=3)
        # Update field state
        new_state = field.update(total_influence)
        logger.debug(f"Computed new state for field {i}.")
        return new_state

    def apply_brush(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        field_index: int,
        value: float,
    ) -> None:
        """
        Applies a brush value to a specific field in the specified area.
        """
        if not (0 <= field_index < self.num_fields):
            raise IndexError(f"Field index out of range. Received field_index={field_index}.")

        # Ensure coordinates are within bounds
        x_min_clamped = max(x_min, 0)
        x_max_clamped = min(x_max, self.width)
        y_min_clamped = max(y_min, 0)
        y_max_clamped = min(y_max, self.height)

        # Apply the brush value and round to 3 decimal places
        self.fields[field_index].state[y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped] = np.round(value, decimals=3)
        logger.info(
            f"Applied brush to field {field_index} at "
            f"x: [{x_min_clamped}, {x_max_clamped}), "
            f"y: [{y_min_clamped}, {y_max_clamped})."
        )

    def get_combined_state(self) -> np.ndarray:
        """
        Returns the combined state of all fields for visualization.
        """
        combined_state = np.sum([field.state for field in self.fields], axis=0)
        max_combined_value = self.num_fields * self.max_value
        min_combined_value = -self.num_fields * self.max_value  # Corrected min value

        # Normalize to [0, 1]
        combined_state_normalized = np.clip(
            (combined_state - min_combined_value) / (max_combined_value - min_combined_value),
            0,
            1,
        )
        # Round to 3 decimal places
        combined_state_normalized = np.round(combined_state_normalized, decimals=3)
        logger.debug("Computed combined state for visualization.")
        return combined_state_normalized

    def get_statistics(self) -> Tuple[float, float, float]:
        """
        Calculates statistics of the combined fields.
        """
        combined_state = self.get_combined_state()
        min_value = np.min(combined_state)
        max_value = np.max(combined_state)
        mean_value = np.mean(combined_state)
        logger.debug(
            f"Computed statistics - Min: {min_value}, Max: {max_value}, Mean: {mean_value}"
        )
        return min_value, max_value, mean_value

    def save_state(self, filename: str) -> None:
        """
        Saves the current state of the simulation to a file using pickle.
        """
        try:
            # Prepare data to save
            data_to_save = {
                "interaction_kernels": self.interaction_kernels,
                "num_fields": self.num_fields,
                "width": self.width,
                "height": self.height,
                "max_value": self.max_value,
                "symmetric_kernels": self.symmetric_kernels,
                "fields": []
            }

            for field in self.fields:
                field_data = {
                    "state": field.state,
                    "activation_params": field.activation_params,
                    "activation_function": field.activation_function_name
                }
                data_to_save["fields"].append(field_data)

            # Open the file in binary write mode and dump the data
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Simulation state saved to {filename}.")
        except Exception as e:
            logger.error(f"Failed to save simulation state to {filename}: {e}")
            raise

    def load_state(self, filename: str) -> None:
        """
        Loads the simulation state from a file using pickle.
        """
        try:
            # Open the file in binary read mode and load the data
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Validate loaded data
            required_keys = {"interaction_kernels", "num_fields", "width", "height", "max_value", "symmetric_kernels", "fields"}
            if not required_keys.issubset(data.keys()):
                missing = required_keys - data.keys()
                raise KeyError(f"Missing keys in the saved state: {missing}")

            # Load simulation parameters
            self.num_fields = int(data["num_fields"])
            self.width = int(data["width"])
            self.height = int(data["height"])
            self.max_value = float(data["max_value"])
            self.symmetric_kernels = bool(data["symmetric_kernels"])

            # Validate compatibility with current simulation
            if self.width != data["width"] or self.height != data["height"]:
                raise ValueError("Loaded state dimensions do not match the current simulation dimensions.")

            # Load interaction_kernels
            self.interaction_kernels = data["interaction_kernels"]

            # Re-initialize fields with loaded attributes
            self.fields = []
            for i in range(self.num_fields):
                field_data = data["fields"][i]
                field = Field(
                    width=self.width,
                    height=self.height,
                    max_value=self.max_value,
                    field_index=i,
                    symmetric_kernels=self.symmetric_kernels,
                )

                # Restore field state and activation parameters
                field.state = field_data["state"]
                field.activation_params = field_data["activation_params"]

                # Set the activation function based on the loaded name
                try:
                    field.set_activation_function(field_data["activation_function"])
                except ValueError as ve:
                    logger.error(
                        f"Error setting activation function for field {i}: {ve}"
                    )
                    raise

                self.fields.append(field)

            logger.info(f"Simulation state loaded from {filename}.")
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except KeyError as ke:
            logger.error(f"Key error during loading state: {ke}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading state from {filename}: {e}")
            raise

    def reset(self, randomize: bool = True, retain_patterns: bool = False) -> None:
        """
        Resets all fields in the simulation and randomizes activation functions and interaction kernels.
        """
        for field in self.fields:
            field.reset_state(randomize=randomize, retain_patterns=retain_patterns)
            # Re-initialize activation parameters and set activation function
            new_activation_params = field.initialize_activation_params()
            field.set_activation_params(*new_activation_params)
        # Re-initialize interaction kernels
        self.interaction_kernels = self.initialize_interaction_kernels()
        logger.info("Simulation has been reset with randomized activation functions and interaction kernels.")

class Button:
    """
    Represents a button in the user interface.
    """
    def __init__(self, rect: Tuple[int, int, int, int], text: str, font: pygame.font.Font,
                 color_normal: Tuple[int, int, int], color_hover: Tuple[int, int, int],
                 text_color: Tuple[int, int, int]):
        """
        Initializes the Button.

        Parameters:
        - rect (Tuple[int, int, int, int]): (x, y, width, height) of the button.
        - text (str): Text displayed on the button.
        - font (pygame.font.Font): Font used for the button text.
        - color_normal (Tuple[int, int, int]): Normal color of the button.
        - color_hover (Tuple[int, int, int]): Color of the button when hovered.
        - text_color (Tuple[int, int, int]): Color of the button text.
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color_normal = color_normal
        self.color_hover = color_hover
        self.text_color = text_color

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the button on the given surface.
        """
        mouse_pos = pygame.mouse.get_pos()
        color = self.color_hover if self.rect.collidepoint(mouse_pos) else self.color_normal
        pygame.draw.rect(surface, color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        """
        Determines if the button was clicked based on the mouse position.

        Parameters:
        - pos (Tuple[int, int]): The (x, y) position of the mouse click.

        Returns:
        - bool: True if the button was clicked, False otherwise.
        """
        return self.rect.collidepoint(pos)

class Visualizer:
    """
    Handles the visualization and user interaction for the simulation.
    """

    def __init__(self, simulation: 'Simulation'):
        """
        Initializes the Visualizer.
        """
        self.simulation = simulation
        self.width = simulation.width
        self.height = simulation.height
        self.max_value = simulation.max_value  # Use the max_value from the simulation

        # Initialize Pygame
        pygame.init()

        # Adjusted window dimensions
        self.control_panel_width = 200  # Width of the control panel on the right
        self.cell_size = CELL_SIZE  # Local cell size
        self.window_width = self.width * self.cell_size + self.control_panel_width
        self.window_height = self.height * self.cell_size

        # Check if window size fits on the screen
        display_info = pygame.display.Info()
        if (self.window_width > display_info.current_w) or (self.window_height > display_info.current_h):
            logger.warning("Window size exceeds screen resolution. Adjusting cell_size.")
            self.cell_size = max(1, min(display_info.current_w // self.width, display_info.current_h // self.height))
            self.window_width = self.width * self.cell_size + self.control_panel_width
            self.window_height = self.height * self.cell_size

        self.surface = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("General Cellular Automaton Simulation")
        self.clock = pygame.time.Clock()

        # Control variables
        self.running = True
        self.playing = False  # Initially paused
        self.iteration = 0

        # Brush control variables
        self.BRUSH_SIZE = 1  # Initial brush size
        self.BRUSH_VALUE = 0.5 * self.max_value  # Start at half of max_value
        self.selected_field = 0  # Field index to apply the brush to

        # Kernel symmetry control variable
        self.symmetric_kernels = simulation.symmetric_kernels

        # Display mode control variable
        self.display_mode: Union[str, int] = 'combined'  # Can be 'combined' or an integer field index

        # Anti-Epilepsy control variables
        self.frame_skip = 0  # Number of frames to skip; 0 means display every frame
        self.frame_counter = 0  # Counter to keep track of skipped frames

        # Font for text
        pygame.font.init()
        self.font_normal = pygame.font.SysFont(None, 20)

        # Precompute the cell surface for rendering
        self.cell_surface = pygame.Surface((self.width, self.height))
        self.cell_array = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Buttons
        self.button_width = 30
        self.button_height = 30

        # Control panel position
        self.controls_x = self.width * self.cell_size + 10
        self.controls_y_start = 10
        self.controls_y = self.controls_y_start

        # Create buttons
        self.create_buttons()

        # Display control variables
        self.display_skip = 1  # Number of iterations to skip between displays
        self.skip_counter = 0

    def create_buttons(self) -> None:
        """
        Creates the interactive buttons.
        """
        # Adjust control positions vertically
        y_offset = 0

        # Brush Size Controls
        self.brush_size_minus_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, self.button_width, self.button_height),
            text="-",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        self.brush_size_plus_button = Button(
            rect=(self.controls_x + 40, self.controls_y + y_offset, self.button_width, self.button_height),
            text="+",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += self.button_height + 10

        # Brush Value Controls
        self.brush_value_minus_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, self.button_width, self.button_height),
            text="-",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        self.brush_value_plus_button = Button(
            rect=(self.controls_x + 40, self.controls_y + y_offset, self.button_width, self.button_height),
            text="+",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += self.button_height + 10

        # Field Selection Button
        self.field_select_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, 150, 30),
            text=f"Field: {self.selected_field}",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += 40

        # Kernel Symmetry Toggle Button
        self.kernel_symmetry_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, 150, 30),
            text=f"Kernels: {'Symmetric' if self.symmetric_kernels else 'Asymmetric'}",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += 40

        # Save and Load Buttons
        self.save_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, 70, 30),
            text="Save",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        self.load_button = Button(
            rect=(self.controls_x + 80, self.controls_y + y_offset, 70, 30),
            text="Load",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += 40

        # Display Mode Button
        self.display_mode_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, 150, 30),
            text="Display: Combined",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += 40

        # Anti-Epilepsy Mode Button
        self.epilepsy_button = Button(
            rect=(self.controls_x, self.controls_y + y_offset, 150, 30),
            text="Anti-Epilepsy: OFF",
            font=self.font_normal,
            color_normal=COLORS["button"],
            color_hover=COLORS["button_hover"],
            text_color=COLORS["text"]
        )
        y_offset += 40

        # Update controls_y_end for dynamic elements
        self.controls_y_end = self.controls_y + y_offset

    def run(self) -> None:
        """
        Runs the main loop of the visualization.
        """
        logger.info("Starting Simulation...")
        try:
            while self.running:
                self.handle_events()
                if self.playing:
                    self.simulation.update()
                    self.iteration += 1

                # Control display update rate
                if self.frame_counter >= self.frame_skip:
                    self.draw()
                    self.frame_counter = 0
                else:
                    self.frame_counter += 1

                # Control update speed
                if self.playing:
                    self.clock.tick(30)  # Limit to 30 FPS when playing
                else:
                    self.clock.tick(60)  # Limit to 60 FPS when paused
        except Exception as e:
            logger.error(f"An error occurred during simulation run: {e}")
        finally:
            pygame.quit()
            logger.info("Pygame quit successfully.")

    def handle_events(self) -> None:
        """
        Handles user input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)
            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event)

    def handle_mouse_click(self, event: pygame.event.Event) -> None:
        """
        Handles mouse click events.
        """
        mouse_pos = pygame.mouse.get_pos()

        # Interaction with the main grid
        if mouse_pos[0] < self.width * self.cell_size and mouse_pos[1] < self.height * self.cell_size:
            cell_x = mouse_pos[0] // self.cell_size
            cell_y = mouse_pos[1] // self.cell_size

            if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                if event.button == 1:
                    # Left Click: Set cell to brush value
                    brush_value = self.BRUSH_VALUE
                elif event.button == 3:
                    # Right Click: Set cell to negative brush value
                    brush_value = -self.BRUSH_VALUE
                else:
                    return  # Ignore other buttons

                # Apply the brush over an area of size BRUSH_SIZE
                half_brush = self.BRUSH_SIZE // 2
                x_min = max(int(cell_x - half_brush), 0)
                x_max = min(int(cell_x + half_brush + 1), self.width)
                y_min = max(int(cell_y - half_brush), 0)
                y_max = min(int(cell_y + half_brush + 1), self.height)

                # Update the selected field at the position
                self.simulation.apply_brush(x_min, x_max, y_min, y_max, self.selected_field, brush_value)
                logger.debug(f"Applied brush at x: [{x_min}, {x_max}), y: [{y_min}, {y_max}) on field {self.selected_field} with value {brush_value}.")

        else:
            # Interaction with buttons
            self.handle_button_click(mouse_pos)

    def handle_button_click(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Determines which button was clicked and performs the corresponding action.
        """
        if self.brush_size_minus_button.is_clicked(mouse_pos):
            if self.BRUSH_SIZE > 1:
                self.BRUSH_SIZE -= BRUSH_SIZE_INCREMENT  # Keep odd for centering
                self.BRUSH_SIZE = max(self.BRUSH_SIZE, 1)
                logger.info(f"Brush size decreased to {self.BRUSH_SIZE}.")
        elif self.brush_size_plus_button.is_clicked(mouse_pos):
            if self.BRUSH_SIZE < MAX_BRUSH_SIZE:
                self.BRUSH_SIZE += BRUSH_SIZE_INCREMENT
                self.BRUSH_SIZE = min(self.BRUSH_SIZE, MAX_BRUSH_SIZE)
                logger.info(f"Brush size increased to {self.BRUSH_SIZE}.")
        elif self.brush_value_minus_button.is_clicked(mouse_pos):
            self.BRUSH_VALUE -= BRUSH_VALUE_INCREMENT * self.max_value
            self.BRUSH_VALUE = max(self.BRUSH_VALUE, -self.max_value)
            self.BRUSH_VALUE = np.round(self.BRUSH_VALUE, decimals=3)
            logger.info(f"Brush value decreased to {self.BRUSH_VALUE:.2f}.")
        elif self.brush_value_plus_button.is_clicked(mouse_pos):
            self.BRUSH_VALUE += BRUSH_VALUE_INCREMENT * self.max_value
            self.BRUSH_VALUE = min(self.BRUSH_VALUE, self.max_value)
            self.BRUSH_VALUE = np.round(self.BRUSH_VALUE, decimals=3)
            logger.info(f"Brush value increased to {self.BRUSH_VALUE:.2f}.")
        elif self.field_select_button.is_clicked(mouse_pos):
            # Cycle through fields
            self.selected_field = (self.selected_field + 1) % self.simulation.num_fields
            self.field_select_button.text = f"Field: {self.selected_field}"
            logger.info(f"Selected field changed to {self.selected_field}.")
        elif self.kernel_symmetry_button.is_clicked(mouse_pos):
            # Toggle kernel symmetry
            self.symmetric_kernels = not self.symmetric_kernels
            self.kernel_symmetry_button.text = f"Kernels: {'Symmetric' if self.symmetric_kernels else 'Asymmetric'}"
            logger.info(f"Kernels symmetry toggled to {'Symmetric' if self.symmetric_kernels else 'Asymmetric'}.")

            # Update the symmetry setting in the simulation and regenerate interaction kernels
            self.simulation.symmetric_kernels = self.symmetric_kernels
            self.simulation.interaction_kernels = self.simulation.initialize_interaction_kernels()
            logger.debug("Regenerated interaction kernels with new symmetry settings.")
        elif self.save_button.is_clicked(mouse_pos):
            filename = 'save_state.pkl'
            try:
                self.simulation.save_state(filename)
                logger.info(f"Simulation state saved to {filename}.")
            except Exception as e:
                logger.error(f"Failed to save simulation state: {e}")
        elif self.load_button.is_clicked(mouse_pos):
            filename = 'save_state.pkl'
            try:
                self.simulation.load_state(filename)
                # Update visualizer parameters
                self.update_visualizer_parameters()
                logger.info(f"Simulation state loaded from {filename}.")
            except Exception as e:
                logger.error(f"Failed to load simulation state: {e}")
        elif self.display_mode_button.is_clicked(mouse_pos):
            # Cycle through display modes
            if self.display_mode == 'combined':
                self.display_mode = 0  # Start with field 0
            else:
                self.display_mode += 1
                if self.display_mode >= self.simulation.num_fields:
                    self.display_mode = 'combined'
            # Update display mode button text
            if self.display_mode == 'combined':
                self.display_mode_button.text = "Display: Combined"
                logger.info("Display mode set to Combined.")
            else:
                self.display_mode_button.text = f"Display: Field {self.display_mode}"
                logger.info(f"Display mode set to Field {self.display_mode}.")
        elif self.epilepsy_button.is_clicked(mouse_pos):
            # Toggle anti-epilepsy mode
            if self.frame_skip == 0:
                self.frame_skip = 1  # Skip every other frame
                self.epilepsy_button.text = "Anti-Epilepsy: ON"
                logger.info("Anti-Epilepsy mode ON: Displaying every other frame.")
            else:
                self.frame_skip = 0
                self.epilepsy_button.text = "Anti-Epilepsy: OFF"
                logger.info("Anti-Epilepsy mode OFF: Displaying every frame.")

    def handle_key_press(self, event: pygame.event.Event) -> None:
        """
        Handles key press events.
        """
        if event.key == pygame.K_SPACE:
            # Toggle play/pause
            self.playing = not self.playing
            logger.info(f"Simulation {'resumed' if self.playing else 'paused'}.")
        elif event.key == pygame.K_RIGHT:
            if not self.playing:
                # Advance one iteration if paused
                self.simulation.update()
                self.iteration += 1
                logger.info("Advanced one iteration.")
        elif event.key == pygame.K_r:
            # Reset the simulation with initial patterns
            logger.info("Resetting simulation with initial patterns...")
            self.simulation.reset(randomize=False, retain_patterns=True)
            self.update_visualizer_parameters()
            self.iteration = 0
        elif event.key == pygame.K_UP:
            # Increase brush size
            if self.BRUSH_SIZE < MAX_BRUSH_SIZE:
                self.BRUSH_SIZE += BRUSH_SIZE_INCREMENT  # Keep odd for centering
                self.BRUSH_SIZE = min(self.BRUSH_SIZE, MAX_BRUSH_SIZE)
                logger.info(f"Brush size increased to {self.BRUSH_SIZE}.")
        elif event.key == pygame.K_DOWN:
            # Decrease brush size
            if self.BRUSH_SIZE > 1:
                self.BRUSH_SIZE -= BRUSH_SIZE_INCREMENT
                self.BRUSH_SIZE = max(self.BRUSH_SIZE, 1)
                logger.info(f"Brush size decreased to {self.BRUSH_SIZE}.")
        elif event.key in [pygame.K_MINUS, pygame.K_KP_MINUS]:
            # Decrease brush value
            self.BRUSH_VALUE -= BRUSH_VALUE_INCREMENT * self.max_value
            self.BRUSH_VALUE = max(self.BRUSH_VALUE, -self.max_value)
            self.BRUSH_VALUE = np.round(self.BRUSH_VALUE, decimals=3)
            logger.info(f"Brush value decreased to {self.BRUSH_VALUE:.2f}.")
        elif event.key in [pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS]:
            # Increase brush value
            self.BRUSH_VALUE += BRUSH_VALUE_INCREMENT * self.max_value
            self.BRUSH_VALUE = min(self.BRUSH_VALUE, self.max_value)
            self.BRUSH_VALUE = np.round(self.BRUSH_VALUE, decimals=3)
            logger.info(f"Brush value increased to {self.BRUSH_VALUE:.2f}.")
        elif event.key == pygame.K_f:
            # Cycle through fields
            self.selected_field = (self.selected_field + 1) % self.simulation.num_fields
            self.field_select_button.text = f"Field: {self.selected_field}"
            logger.info(f"Selected field changed to {self.selected_field}.")
        elif event.key == pygame.K_k:
            # Toggle kernel symmetry with 'K' key
            self.symmetric_kernels = not self.symmetric_kernels
            self.kernel_symmetry_button.text = f"Kernels: {'Symmetric' if self.symmetric_kernels else 'Asymmetric'}"
            logger.info(f"Kernels symmetry toggled to {'Symmetric' if self.symmetric_kernels else 'Asymmetric'}.")

            # Update the symmetry setting in the simulation and regenerate interaction kernels
            self.simulation.symmetric_kernels = self.symmetric_kernels
            self.simulation.interaction_kernels = self.simulation.initialize_interaction_kernels()
            logger.debug("Regenerated interaction kernels with new symmetry settings.")
        elif event.key == pygame.K_s:
            # Save simulation state
            filename = 'save_state.pkl'
            try:
                self.simulation.save_state(filename)
                logger.info(f"Simulation state saved to {filename}.")
            except Exception as e:
                logger.error(f"Failed to save simulation state: {e}")
        elif event.key == pygame.K_l:
            # Load simulation state
            filename = 'save_state.pkl'
            try:
                self.simulation.load_state(filename)
                # Update visualizer parameters
                self.update_visualizer_parameters()
                logger.info(f"Simulation state loaded from {filename}.")
            except Exception as e:
                logger.error(f"Failed to load simulation state: {e}")
        elif event.key == pygame.K_d:
            # Cycle through display modes
            if self.display_mode == 'combined':
                self.display_mode = 0  # Start with field 0
            else:
                self.display_mode += 1
                if self.display_mode >= self.simulation.num_fields:
                    self.display_mode = 'combined'
            # Update display mode button text
            if self.display_mode == 'combined':
                self.display_mode_button.text = "Display: Combined"
                logger.info("Display mode set to Combined.")
            else:
                self.display_mode_button.text = f"Display: Field {self.display_mode}"
                logger.info(f"Display mode set to Field {self.display_mode}.")
        elif event.key == pygame.K_e:
            # Toggle anti-epilepsy mode
            if self.frame_skip == 0:
                self.frame_skip = 1  # Skip every other frame
                self.epilepsy_button.text = "Anti-Epilepsy: ON"
                logger.info("Anti-Epilepsy mode ON: Displaying every other frame.")
            else:
                self.frame_skip = 0
                self.epilepsy_button.text = "Anti-Epilepsy: OFF"
                logger.info("Anti-Epilepsy mode OFF: Displaying every frame.")

    def update_visualizer_parameters(self) -> None:
        """
        Updates visualizer parameters after loading a simulation state.
        """
        self.width = self.simulation.width
        self.height = self.simulation.height
        self.max_value = self.simulation.max_value
        self.symmetric_kernels = self.simulation.symmetric_kernels
        self.selected_field = 0  # Reset selected field
        self.display_mode = 'combined'  # Reset display mode

        # Update window size if necessary
        self.window_width = self.width * self.cell_size + self.control_panel_width
        self.window_height = self.height * self.cell_size
        self.surface = pygame.display.set_mode((self.window_width, self.window_height))

        # Recreate buttons to position them correctly
        self.create_buttons()

        # Recreate cell surface
        self.cell_surface = pygame.Surface((self.width, self.height))
        self.cell_array = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        logger.debug("Visualizer parameters updated after loading simulation state.")

    def draw(self) -> None:
        """
        Draws all elements on the screen.
        """
        # Draw background and grid
        self.surface.fill(COLORS["background"])
        draw_grid(self.surface, self.width, self.height, self.cell_size)

        # Draw cells
        self.draw_cells()

        # Draw control panel
        self.draw_controls()

        # Update display
        pygame.display.flip()

    def draw_cells(self) -> None:
        """
        Draws the cells on the Pygame surface based on the current display mode.
        """
        try:
            if self.display_mode == 'combined':
                combined_state = self.simulation.get_combined_state()
                colors = self.map_values_to_color(combined_state, mode='combined')
            else:
                field_index = self.display_mode
                if not (0 <= field_index < self.simulation.num_fields):
                    logger.error(f"Invalid field index {field_index} for display.")
                    return
                field_state = self.simulation.fields[field_index].state
                # Normalize to [-1, 1]
                normalized_state = np.clip(field_state / self.max_value, -1, 1)
                colors = self.map_values_to_color(normalized_state, mode='individual')

            # Update the cell array
            self.cell_array[:, :, :] = colors
            # Convert the cell array to a surface
            pygame.surfarray.blit_array(self.cell_surface, self.cell_array.swapaxes(0, 1))
            # Scale the cell surface to the desired size
            scaled_surface = pygame.transform.scale(self.cell_surface, (self.width * self.cell_size, self.height * self.cell_size))
            # Blit the scaled surface onto the main surface
            self.surface.blit(scaled_surface, (0, 0))
            logger.debug("Cells drawn on the grid.")
        except Exception as e:
            logger.error(f"Error drawing cells: {e}")

    def map_values_to_color(self, state: np.ndarray, mode: str = 'individual') -> np.ndarray:
        """
        Maps numerical field values to RGB colors for visualization.
        """
        if mode == 'combined':
            # Normalize combined state to range [-1, 1]
            normalized_state = state * 2 - 1  # Assuming get_combined_state returns [0,1], map to [-1,1]

            # Initialize color channels
            red = np.zeros_like(state)
            green = np.zeros_like(state)
            blue = np.zeros_like(state)

            # Masks
            pos_mask = normalized_state > 0
            neg_mask = normalized_state < 0
            zero_mask = normalized_state == 0

            # Define color gradients
            # Negative: Blue -> Dark Blue -> Violet -> Black
            neg_colors = np.array([

                (180, 170, 210), # Light Pale Blue-Grey
                (160, 150, 190), # Light Blue-Grey
                (140, 130, 170), # Pale Blue
                (120, 110, 150), # Soft Blue-Grey
                (100, 90, 130),  # Light Blue-Purple
                (80, 70, 110),   # Medium Blue
                (60, 50, 90),    # Deep Blue
                (40, 30, 70),    # Dark Blue-Grey
                (20, 10, 50),    # Dark Blue-Purple
                (0, 0, 30),      # Very Dark Blue
                (0, 0, 0)        # Black (para el valor 0)


            ])
            neg_x = np.linspace(-1, 0, len(neg_colors))
            # Map negative values
            neg_normalized = normalized_state[neg_mask]
            red[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 0])
            green[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 1])
            blue[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 2])

            # Positive: Black -> Dark Grey -> Light Grey -> Red -> Yellow -> White
            pos_colors = np.array([
            
                (0, 0, 0),       # Black (para el valor 0)
                (40, 40, 40),    # Very Dark Grey
                (80, 80, 80),    # Dark Grey
                (120, 120, 120), # Mid Grey
                (160, 160, 160), # Light Grey
                (200, 60, 60),   # Dark Red
                (210, 90, 50),   # Deep Red-Orange
                (220, 120, 40),  # Red-Orange
                (230, 150, 30),  # Bright Orange
                (240, 180, 20),  # Bright Orange-Yellow
                (250, 210, 10)   # Bright Yellow
        

            ])
            pos_x = np.linspace(0, 1, len(pos_colors))
            # Map positive values
            pos_normalized = normalized_state[pos_mask]
            red[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 0])
            green[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 1])
            blue[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 2])

            # Zero values mapped to Black
            red[zero_mask], green[zero_mask], blue[zero_mask] = 0, 0, 0

        elif mode == 'individual':
            # For individual fields, handle negative and positive separately
            red, green, blue = np.zeros_like(state), np.zeros_like(state), np.zeros_like(state)
            pos_mask = state > 0
            neg_mask = state < 0
            zero_mask = state == 0

            pos_normalized = np.clip(state[pos_mask] / self.max_value, 0, 1)
            neg_normalized = np.clip(-state[neg_mask] / self.max_value, 0, 1)

            # Define color gradients
            # Positive: White -> Red
            pos_colors = np.array([
                (255, 255, 255),  # White
                (255, 0, 0),      # Red
            ])
            pos_x = np.linspace(0, 1, len(pos_colors))
            # Map positive values
            red[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 0])
            green[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 1])
            blue[pos_mask] = np.interp(pos_normalized, pos_x, pos_colors[:, 2])

            # Negative: White -> Blue
            neg_colors = np.array([
                (255, 255, 255),  # White
                (0, 0, 255),      # Blue
            ])
            neg_x = np.linspace(0, 1, len(neg_colors))
            # Map negative values
            red[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 0])
            green[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 1])
            blue[neg_mask] = np.interp(neg_normalized, neg_x, neg_colors[:, 2])

            # Zero values mapped to White
            red[zero_mask], green[zero_mask], blue[zero_mask] = 255, 255, 255

        else:
            logger.error(f"Invalid mode '{mode}' for color mapping.")
            red = green = blue = np.zeros_like(state)

        colors = np.stack([red, green, blue], axis=-1).astype(np.uint8)
        return colors

    def draw_controls(self) -> None:
        """
        Draws the UI controls and text on the control panel.
        """
        try:
            # Draw control panel background
            control_panel_rect = pygame.Rect(self.width * self.cell_size, 0, self.control_panel_width, self.window_height)
            pygame.draw.rect(self.surface, COLORS["background"], control_panel_rect)

            # Draw brush size buttons and labels
            self.brush_size_minus_button.draw(self.surface)
            self.brush_size_plus_button.draw(self.surface)
            brush_size_label = self.font_normal.render("Brush Size:", True, COLORS["text"])
            self.surface.blit(brush_size_label, (self.controls_x + 80, self.controls_y_start))
            brush_size_value = self.font_normal.render(f"{self.BRUSH_SIZE}", True, COLORS["text"])
            self.surface.blit(brush_size_value, (self.controls_x + 80, self.controls_y_start + 20))

            # Draw brush value buttons and labels
            self.brush_value_minus_button.draw(self.surface)
            self.brush_value_plus_button.draw(self.surface)
            brush_value_label = self.font_normal.render("Brush Value:", True, COLORS["text"])
            self.surface.blit(brush_value_label, (self.controls_x + 80, self.controls_y_start + 50))
            brush_value_display = self.font_normal.render(f"{self.BRUSH_VALUE:.2f}", True, COLORS["text"])
            self.surface.blit(brush_value_display, (self.controls_x + 80, self.controls_y_start + 70))

            # Draw field selection button
            self.field_select_button.draw(self.surface)

            # Draw kernel symmetry button
            self.kernel_symmetry_button.draw(self.surface)

            # Draw save and load buttons
            self.save_button.draw(self.surface)
            self.load_button.draw(self.surface)

            # Draw display mode button
            self.display_mode_button.draw(self.surface)

            # Draw Anti-Epilepsy Mode button
            self.epilepsy_button.draw(self.surface)

            # Display status and instructions
            status_text = self.font_normal.render(f"Iteration: {self.iteration} - {'Playing' if self.playing else 'Paused'}", True, COLORS["text"])
            instructions_text = self.font_normal.render(
                "Space: Play/Pause | Right Arrow: Step | R: Reset",
                True,
                COLORS["text"]
            )
            instructions_text2 = self.font_normal.render(
                "F: Select Field | K: Toggle Kernels | D: Change Display",
                True,
                COLORS["text"]
            )
            instructions_text3 = self.font_normal.render(
                "S: Save | L: Load | E: Toggle Anti-Epilepsy",
                True,
                COLORS["text"]
            )
            brush_info_text = self.font_normal.render("Mouse Click: Paint | +/-: Brush Value | Up/Down: Brush Size", True, COLORS["text"])

            instructions_y = self.controls_y_end + 10
            self.surface.blit(status_text, (self.controls_x, instructions_y))
            self.surface.blit(instructions_text, (self.controls_x, instructions_y + 20))
            self.surface.blit(instructions_text2, (self.controls_x, instructions_y + 40))
            self.surface.blit(instructions_text3, (self.controls_x, instructions_y + 60))
            self.surface.blit(brush_info_text, (self.controls_x, instructions_y + 80))

            # Add statistics
            min_value, max_value, mean_value = self.simulation.get_statistics()
            stats_text = self.font_normal.render(f"Min: {min_value:.2f} | Max: {max_value:.2f} | Mean: {mean_value:.2f}", True, COLORS["text"])
            self.surface.blit(stats_text, (self.controls_x, instructions_y + 100))

            logger.debug("Control panel drawn.")
        except Exception as e:
            logger.error(f"Error drawing controls: {e}")

def main():
    """
    Main function to run the cellular automaton simulation.
    """
    # Define grid size
    grid_width = 250  # Number of cells horizontally
    grid_height = 250  # Number of cells vertically

    # Define number of fields
    num_fields = 5

    # Define maximum cell value
    max_value = 10.0

    # Define kernel symmetry
    symmetric_kernels = True

    # Initialize Simulation
    try:
        simulation = Simulation(
            width=grid_width,
            height=grid_height,
            num_fields=num_fields,
            max_value=max_value,
            symmetric_kernels=symmetric_kernels
        )
        logger.info("Simulation initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Simulation: {e}")
        sys.exit(1)

    # Initialize Visualizer
    try:
        visualizer = Visualizer(simulation)
        logger.info("Visualizer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Visualizer: {e}")
        sys.exit(1)

    # Run the simulation visualization
    visualizer.run()

if __name__ == "__main__":
    main()
