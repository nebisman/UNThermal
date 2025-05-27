
# Unthermal

Unthermal is a Python package designed to facilitate the learning and understanding of control systems. This package provides a suite of tools and examples that help students and educators simulate and analyze various control system dynamics and responses.

## Features

- **Simulation Tools**: Simulate the behavior of various control systems under different conditions.
- **Analysis Utilities**: Analyze stability, frequency response, and other key metrics of control systems.
- **Interactive Examples**: Hands-on, interactive examples to aid in understanding complex concepts.
- **Educational Materials**: Access to tutorials and guides for using the package in educational settings.

## Installation

You can install `unthermal` directly using pip:

\`\`\`bash
pip install unthermal
\`\`\`

## Quick Start

Here’s a quick example to get you started with `unthermal`:

\`\`\`python
from unthermal import Controller, System

# Define your system parameters
system = System(type='thermal', parameters={'gain': 1.0, 'time_constant': 5})

# Create a controller
controller = Controller(type='PID', parameters={'P': 0.5, 'I': 0.1, 'D': 0.05})

# Analyze the system
response = controller.analyze(system)
print(response)
\`\`\`

## Documentation

For full documentation, including tutorials and a detailed API reference, visit [Unthermal Documentation](https://unthermal.readthedocs.io).

## Contributing

We welcome contributions from the community, including bug reports, feature requests, and code contributions. For more information on how to contribute, please see the `CONTRIBUTING.md` file in our repository.

## License

`unthermal` is open source and licensed under the MIT license. See the `LICENSE` file for more details.

## Support

If you have any questions or issues, please open an issue on our GitHub repository, or contact us directly at support@unthermal.com.

## Authors and Acknowledgment

Unthermal was developed by a dedicated team of educators and engineers passionate about improving control systems education. We thank our contributors and the community for their ongoing support.

## See Also

- [Control Systems Library](https://python-control.readthedocs.io)
- [Matplotlib](https://matplotlib.org) for plotting support
