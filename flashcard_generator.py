import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import PyPDF2
import io
import json
import csv
from typing import List, Dict

class FlashcardGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, max_chunk_size: int = 800) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_flashcards_from_chunk(self, text_chunk: str, subject: str = "") -> List[Dict[str, str]]:
        subject_context = f"for {subject} subject" if subject else ""
        
        prompt = f"""Generate educational flashcards {subject_context} from the following text. Create question-answer pairs that test understanding of key concepts, facts, and relationships. Format each flashcard as 'Q: [question] A: [answer]'.

Text: {text_chunk}

Flashcards:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=400,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self.parse_flashcards(generated_text)
    
    def parse_flashcards(self, generated_text: str) -> List[Dict[str, str]]:
        flashcards = []
        
        parts = generated_text.split('Q:')
        
        for part in parts[1:]: 
            if 'A:' in part:
                question_answer = part.strip()
                
                qa_parts = question_answer.split('A:', 1)
                if len(qa_parts) == 2:
                    question = qa_parts[0].strip()
                    answer = qa_parts[1].strip()
                    
                    answer = answer.split('Q:')[0].strip()
                    
                    if question and answer and len(question) > 5 and len(answer) > 5:
                        flashcards.append({
                            'question': question,
                            'answer': answer
                        })
        
        return flashcards
    
    def enhance_flashcards(self, flashcards: List[Dict[str, str]]) -> List[Dict[str, str]]:
        enhanced_flashcards = []
        
        for card in flashcards:
            question_length = len(card['question'])
            answer_length = len(card['answer'])
            
            if question_length < 50 and answer_length < 100:
                difficulty = "Easy"
            elif question_length < 100 and answer_length < 200:
                difficulty = "Medium"
            else:
                difficulty = "Hard"
            
            enhanced_card = {
                'question': card['question'],
                'answer': card['answer'],
                'difficulty': difficulty
            }
            enhanced_flashcards.append(enhanced_card)
        
        return enhanced_flashcards
    
    def generate_flashcards(self, text: str, subject: str = "", min_cards: int = 15) -> List[Dict[str, str]]:
        chunks = self.chunk_text(text)
        all_flashcards = []
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                chunk_flashcards = self.generate_flashcards_from_chunk(chunk, subject)
                all_flashcards.extend(chunk_flashcards)
        
        enhanced_flashcards = self.enhance_flashcards(all_flashcards)
        
        if len(enhanced_flashcards) < min_cards:

            larger_chunks = self.chunk_text(text, max_chunk_size=1200)
            for chunk in larger_chunks[:2]: 
                additional_cards = self.generate_flashcards_from_chunk(chunk, subject)
                enhanced_flashcards.extend(self.enhance_flashcards(additional_cards))
        
        unique_flashcards = []
        seen_questions = set()
        
        for card in enhanced_flashcards:
            question_lower = card['question'].lower()
            if question_lower not in seen_questions:
                seen_questions.add(question_lower)
                unique_flashcards.append(card)
        
        return unique_flashcards[:min_cards] if len(unique_flashcards) > min_cards else unique_flashcards

def export_to_csv(flashcards: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'answer', 'difficulty']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for card in flashcards:
            writer.writerow(card)

def export_to_json(flashcards: List[Dict[str, str]], filename: str):
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(flashcards, jsonfile, indent=2, ensure_ascii=False)

def export_to_anki(flashcards: List[Dict[str, str]], filename: str):
    with open(filename, 'w', encoding='utf-8') as ankifile:
        for card in flashcards:
            ankifile.write(f"{card['question']}\t{card['answer']}\n")

def main():
    st.set_page_config(
        page_title="LLM Flashcard Generator",
        layout="wide"
    )
    
    st.title("LLM-Powered Flashcard Generator")
    
    if 'generator' not in st.session_state:
        with st.spinner("Loading Flan-T5 model..."):
            st.session_state.generator = FlashcardGenerator()
    
    st.sidebar.header("Options")
    subject = st.sidebar.selectbox(
        "Subject (optional)",
        ["", "Biology", "Chemistry", "Physics", "History", "Computer Science", "Mathematics", "Literature", "Psychology"]
    )
    
    min_cards = st.sidebar.slider("Minimum flashcards", 10, 30, 10)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Content")
        
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
        
        text_content = ""
        
        if input_method == "Text Input":
            text_content = st.text_area(
                "Paste your content here:",
                height=300,
                placeholder="Enter your lecture notes, or any educational material..."
            )
        
        else:
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=['txt', 'pdf'],
                help="Upload a .txt or .pdf file"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    text_content = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    text_content = st.session_state.generator.extract_text_from_pdf(uploaded_file)
                
                st.success(f"File uploaded successfully!")
        
        generate_button = st.button("Generate Flashcards", type="primary")
    
    with col2:
        st.header("Generated Flashcards")
        
        if generate_button and text_content:
            if len(text_content.strip()) < 100:
                st.error("Please provide more content (at least 100 characters) for better flashcard generation.")
            else:
                with st.spinner("Generating flashcards with Flan-T5..."):
                    flashcards = st.session_state.generator.generate_flashcards(
                        text_content, 
                        subject, 
                        min_cards
                    )
                
                if flashcards:
                    st.success(f"Generated {len(flashcards)} flashcards!")
                    
                    st.session_state.flashcards = flashcards
                    
                    for i, card in enumerate(flashcards, 1):
                        with st.expander(f"Flashcard {i} - {card.get('difficulty', 'Medium')}"):
                            st.write("**Question:**")
                            st.write(card['question'])
                            st.write("**Answer:**")
                            st.write(card['answer'])
                else:
                    st.error("Could not generate flashcards. Please check your input content.")
    
    if 'flashcards' in st.session_state and st.session_state.flashcards:
        st.header("Export Flashcards")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                csv_data = io.StringIO()
                fieldnames = ['question', 'answer', 'difficulty']
                writer = csv.DictWriter(csv_data, fieldnames=fieldnames)
                writer.writeheader()
                for card in st.session_state.flashcards:
                    writer.writerow(card)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data.getvalue(),
                    file_name="flashcards.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export as JSON"):
                json_data = json.dumps(st.session_state.flashcards, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="flashcards.json",
                    mime="application/json"
                )
        
    
    st.markdown("---")
    st.markdown("Built by Shivam")

if __name__ == "__main__":
    main()