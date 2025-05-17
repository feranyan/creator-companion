import sys
import mss
from PIL import Image
import os
from dotenv import load_dotenv
from google import genai

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QDialog,
    QPushButton, QProgressDialog
)
from PyQt6.QtGui import QPainter, QColor, QPen, QGuiApplication, QBrush
from PyQt6.QtCore import (
    Qt, QPoint, QRect, QThread, pyqtSignal, QTimer,
    QMutex, QWaitCondition
)

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# Leia da variável de ambiente:
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final 
def call_agent(agent: Agent, message_text: str) -> str:
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    content = types.Content(role="user", parts=[types.Part(text=message_text)])
    final_response = ""
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response

def agent_translator(original_text):
    translator = Agent(
        name="agent_translator",
        model="gemini-2.0-flash",
        instruction="""
            Você é um experiente tradutor.
            Você recebe um texto e traduz para o português.
            Sua resposta deve ser apenas a tradução em português, sem formatação ou texto adicional.
            """,
        description="Translate texts"
    )
    translated_text = call_agent(translator, original_text)
    return translated_text
    
def agent_analyser(original_text):
    analyser = Agent(
        name="agent_analyser",
        model="gemini-2.0-flash",
        instruction="""
        Você é um analisador de contextos.
        Você recebe um texto e analisa se existem palavras chaves, neste contexto do texto. Você extrai as palavras chaves e as mostra como resposta (somente as palavras chaves).
        Se você conseguir encontrar alguma referência relevante com essas palavras chaves, descreva em português brasileiro os fatos relacionados.
        Se não encontrar nada relevante, responda apenas "[SEM ANÁLISE RELEVANTE]".
        """,
        description="Analyses texts"
    )
    analysed_text = call_agent(analyser, original_text)
    return analysed_text

class ResultDisplayWindow(QDialog):
    def __init__(self, result_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultado da Análise da Imagem")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        title_label = QLabel("Texto Extraído e Traduzido:")
        layout.addWidget(title_label)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(result_text)
        layout.addWidget(self.text_edit)
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)

class LoadingDialog(QDialog):
    def __init__(self, message="Carregando...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aguarde")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(True)
        self.setGeometry(0, 0, 250, 100)
        layout = QVBoxLayout()
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        self.setLayout(layout)
        screen_geo = QGuiApplication.primaryScreen().geometry()
        x = screen_geo.center().x() - self.width() // 2
        y = screen_geo.center().y() - self.height() // 2
        self.move(x, y)
    def setMessage(self, message):
        self.message_label.setText(message)

class CaptureAnalyzeThread(QThread):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, monitor_area, api_key):
        super().__init__()
        self.monitor_area = monitor_area
        self.api_key = api_key
        self._is_running = True


    def run(self):
        img = None
        try:
            print(f"Thread de captura: Capturando área {self.monitor_area}")
            with mss.mss() as sct:
                sct_img = sct.grab(self.monitor_area)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
            print("Thread de captura: Captura bem sucedida.")
        except Exception as e:
            error_msg = f"Erro durante a captura: {e}"
            print(f"Thread de captura: {error_msg}")
            self.error_occurred.emit(error_msg)
            return

        api_result_text = "Erro ao obter resultado da API."
        if img:
            print("Thread da API: Enviando imagem para análise...")
            try:
                if not self.api_key:
                    api_result_text = "Erro da API: A variável de ambiente 'GOOGLE_API_KEY' não foi configurada."
                else:
                    client = genai.Client(api_key=self.api_key)
                    prompt_text = "Extraia o texto da imagem."
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[img, prompt_text]
                    )
                    print("Thread da API: Resposta recebida.")
                    if hasattr(response, 'text'):
                        api_result_text = f"Original: {response.text}"
                        print(response.text)
                        print("Thread da API: Texto extraído com sucesso.")
                        translated_text_agent = agent_translator(response.text)
                        print(translated_text_agent)
                        api_result_text += f"\nTradução: {translated_text_agent}"
                        analysed_text_agent = agent_analyser(response.text)
                        api_result_text += f"\n\nPalavras chave: {analysed_text_agent}"
                        print(analysed_text_agent)
                    else:
                        api_result_text = "Não foi possível extrair texto ou a resposta da API não contém texto."
                        print("Thread da API: Falha na extração de texto.")
            except Exception as e:
                api_result_text = f"Ocorreu um erro ao chamar a API Google Generative AI: {e}\nVerifique sua chave de API, conexão com a internet, e se a API está habilitada para o modelo 'gemini-2.0-flash'."
                print(f"Thread da API: {api_result_text}")

        self.result_ready.emit(api_result_text)
        print("Thread: Emissão de resultado concluída.")


class SelectionWindow(QWidget):
    selection_finished = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.selection_in_progress = False
        print("Janela de seleção pronta. Clique e arraste para selecionar a área.")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.selection_in_progress = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.selection_in_progress and event.buttons() & Qt.MouseButton.LeftButton:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.selection_in_progress:
            self.end_point = event.pos()
            self.selection_in_progress = False
            capture_rect = QRect(self.start_point, self.end_point).normalized()
            self.close()

            if capture_rect.width() > 0 and capture_rect.height() > 0:
                 self.selection_finished.emit(capture_rect)
            else:
                 print("Área de seleção inválida (tamanho zero).")


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 2))
        pen = QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        brush_color = QColor(0, 0, 255, 50)
        brush = QBrush(brush_color)
        painter.setBrush(brush)
        if self.start_point != self.end_point or self.selection_in_progress:
            selection_rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(selection_rect)


# --- Função Principal e Gerenciamento do Fluxo ---
if __name__ == '__main__':
    app = QApplication(sys.argv)

    if not GOOGLE_API_KEY:
        print("Variável de ambiente GOOGLE_API_KEY não configurada. Saindo.")
        sys.exit(1)

    selection_window = SelectionWindow()

    worker_thread = None
    loading_dialog_instance = None

    def start_analysis_thread(capture_rect):
        global worker_thread, loading_dialog_instance

        print("Main: Sinal selection_finished recebido.")

        monitor_area = {
            "top": capture_rect.y(),
            "left": capture_rect.x(),
            "width": capture_rect.width(),
            "height": capture_rect.height()
        }

        # Exibe o diálogo de carregamento
        loading_dialog_instance = LoadingDialog("Analisando...")
        loading_dialog_instance.show() 

        # Cria e inicia a thread de trabalho
        worker_thread = CaptureAnalyzeThread(monitor_area, GOOGLE_API_KEY)

        # --- Conectar sinais da thread de trabalho ---
        def handle_analysis_result(result_text):
            global loading_dialog_instance 

            print("Main: Sinal result_ready recebido.")
            # Fecha o diálogo de carregamento
            if loading_dialog_instance:
                 loading_dialog_instance.accept() 
                 loading_dialog_instance = None 

            # Exibe a janela de resultados
            result_window = ResultDisplayWindow(result_text)
            result_window.exec() 

        def handle_analysis_error(error_message):
             global loading_dialog_instance 

             print("Main: Sinal error_occurred recebido.")
             # Fecha o diálogo de carregamento
             if loading_dialog_instance:
                 loading_dialog_instance.accept()
                 loading_dialog_instance = None

             # Exibe a mensagem de erro na janela de resultados
             error_window = ResultDisplayWindow(f"Ocorreu um erro:\n{error_message}")
             error_window.exec()

        worker_thread.result_ready.connect(handle_analysis_result)
        worker_thread.error_occurred.connect(handle_analysis_error)

        # Conecta a thread para que ela seja deletada automaticamente quando terminar
        worker_thread.finished.connect(worker_thread.deleteLater)

        worker_thread.start()
        print("Main: Thread de análise iniciada.")


    # Conecta o sinal da janela de seleção ao slot no main
    selection_window.selection_finished.connect(start_analysis_thread)


    selection_window.showFullScreen()

    sys.exit(app.exec())