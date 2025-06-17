# LLM-Powered Flashcard Generator

A flashcard generation tool that uses Google's Flan-T5 language model to automatically convert educational content into question-answer flashcards.

## Features

- **LLM Integration**: Uses Flan-T5 for flashcard generation
- **Multiple Input Methods**: Text input or file upload (.txt, .pdf)
- **Subject-Aware Generation**: Optional subject selection for context-aware flashcards
- **Difficulty Levels**: Automatically assigns Easy/Medium/Hard difficulty levels
- **Export Options**: CSV and JSON format
- **Clean UI**: Streamlit-based web interface
- **Batch Processing**: Handles large documents by chunking content

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shrishivam2811/flashcard-generator.git
cd flashcard-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run flashcard_generator.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Input Content**: 
   - Choose between text input or file upload
   - Paste educational content or upload .txt/.pdf files
   - Optionally select a subject for better context

2. **Generate Flashcards**:
   - Click "Generate Flashcards" 
   - Wait for Flan-T5 to process your content
   - Review generated flashcards

3. **Export**:
   - Choose from CSV or JSON formats

## Technical Architecture

### Core Components

1. **FlashcardGenerator Class**: Main class handling LLM integration and flashcard generation
2. **Text Processing**: Chunking algorithm for handling large documents
3. **Export Functions**: Multiple format support for flashcard export
4. **Streamlit UI**: Clean, user-friendly web interface

### Model Details

- **Model**: google/flan-t5-base
- **Framework**: Hugging Face Transformers
- **Input Processing**: Intelligent text chunking (800 char chunks)
- **Generation Parameters**: Temperature=0.7, max_length=400


### Generation Parameters
Adjust generation parameters in the `generate_flashcards_from_chunk` method:

```python
outputs = self.model.generate(
    inputs.input_ids,
    max_length=400,        # Increase for longer answers
    temperature=0.7,       # Adjust for creativity vs consistency
    do_sample=True,        # Enable sampling for variety
    num_return_sequences=1 # Generate multiple versions
)
```

## Performance Notes

- **Model Loading**: First run takes longer due to model download (~1GB)
- **Generation Time**: ~3-5 minutes on CPU, faster on GPU
  
## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure stable internet connection for initial download
2. **Memory Issues**: Use `google/flan-t5-small` for lower memory usage
3. **Poor Quality Output**: Try longer input text or more specific subject selection

## Acknowledgments

- Google Research for Flan-T5 model
- Hugging Face for transformer library
- Streamlit for the web framework
