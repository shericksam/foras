
# Foras: GPT-3 Enhanced Confluence Query Tool
<p align="center">
  <img src="https://github.com/shericksam/foras/blob/main/foras.jpeg" alt="foras" width="200" height="270"/>
</p>
Welcome to Foras, a cutting-edge tool developed by Erick Guerrero to enrich the experience of querying Atlassian Confluence documentation. By leveraging the power of OpenAI's GPT-3, Foras provides users with an intuitive way to find information quickly and efficiently through natural language processing.

## Features

- **Natural Language Processing**: Ask Foras questions as if you were talking to a human and get precise answers from your Confluence workspace.
- **Efficient Information Retrieval**: Dramatically cut down the time you spend sifting through documents by receiving instant, accurate responses.
- **Simple User Interface**: A clean, user-friendly chat interface that makes it easy to interact with your documentation.
- **AI Learning**: Foras gets smarter over time, learning from each query to improve future responses.
- **Future Image Generation for Diagrams**: We're working on integrating a feature to allow Foras to generate visual representations and diagrams directly from your queries.

## Getting Started

### Prerequisites

- An account and API access for Atlassian Confluence.
- Access to OpenAI's GPT-3 API.
- Python 3.8 installed on your machine.

### Installation

1. Clone the repository:
git clone https://github.com/shericksam/foras.git


2. Navigate to the Foras directory:
cd foras


3. Install the required dependencies:
`pip install -r requirements.txt`


### Configuration

- Create a `.env` file from the `.env.example` template.
- Add your `CONFLUENCE_API_TOKEN`, `CONFLUENCE_HOST`, and `OPENAI_API_KEY` to the `.env` file.

### Running Foras

Execute the following command to start the Django server:
`python3.8 manage.py runserver`


You can now access the Foras interface through your browser at the address provided by the Django server.

## Contribution

If you're interested in contributing to Foras, feel free to fork the repository, make your changes, and submit a pull request. For any suggestions or features you'd like to discuss, please open an issue.

## Roadmap

- [x] Query Confluence documentation using GPT-3.
- [x] Include user feedback mechanisms for AI enhancement.
- [ ] Implement image generation for diagram creation.
- [ ] Extend the tool's capabilities to other documentation platforms.

## License

Foras is made available under the MIT License. For more details, see the [LICENSE](LICENSE) file in the repository.

## Contact

Feel free to reach out to Erick Guerrero for any questions or potential collaborations:
- **Email**: [magnus_22_10@hotmail.com](mailto:magnus_22_10@hotmail.com)
- **GitHub**: [@shericksam](https://github.com/shericksam)

## Acknowledgements

A heartfelt thank you to all those who have contributed to this project, provided valuable feedback, and supported the development process. Special thanks to OpenAI for the GPT-3 technology that powers the core functionality of Foras.
