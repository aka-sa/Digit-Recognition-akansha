import sys
from io import BytesIO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QImage, QColor, QFont, QKeySequence
from PyQt5.QtCore import Qt, QPoint
from PIL import Image, ImageDraw, ImageFont
import base64
from basecode import llm  # Ensure this module is correctly set up

class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(1200, 800)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = Qt.white
        self.pen_width = 5
        self.actions = []  # List of actions for undo functionality
        self.current_action = []  # Current action being drawn

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.current_action = []

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.current_action.append((self.last_point, event.pos()))
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_action:
                self.actions.append(self.current_action)
                self.current_action = []

    def clear_canvas(self):
        self.image.fill(Qt.black)
        self.actions.clear()
        self.update()

    def undo_last_action(self):
        if not self.actions:
            return
        self.actions.pop()
        self.redraw_all()

    def redraw_all(self):
        self.image.fill(Qt.black)
        painter = QPainter(self.image)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        for action in self.actions:
            for start, end in action:
                painter.drawLine(start, end)
        self.update()

    def get_pil_image(self):
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        self.image.save(buffer, "PNG")
        pil_image = Image.open(BytesIO(buffer.data()))
        return pil_image

    def get_base64_image(self):
        buffer = BytesIO()
        self.image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_last_equals_position(self):
        """
        Placeholder function to find the position of the last equals sign.
        This should be implemented based on how equals signs are drawn/stored.
        For simplicity, we'll return the center of the canvas.
        """
        return QPoint(self.width() // 2, self.height() // 2)


class DrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Math Notes")
        self.setFixedSize(1200, 900)  # Extra space for buttons

        self.canvas = DrawingCanvas(self)

        # Initialize PIL image for drawing answers
        self.pil_image = Image.new("RGB", (1200, 800), (0, 0, 0))
        self.draw_pil = ImageDraw.Draw(self.pil_image)

        # Set up the layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add canvas
        main_layout.addWidget(self.canvas)

        # Add buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        self.button_clear = QPushButton("Clear")
        self.button_clear.clicked.connect(self.clear_canvas)
        button_layout.addWidget(self.button_clear)

        self.button_undo = QPushButton("Undo (Ctrl+Z)")
        self.button_undo.clicked.connect(self.undo)
        button_layout.addWidget(self.button_undo)

        self.button_calculate = QPushButton("Calculate (Return/Enter)")
        self.button_calculate.clicked.connect(self.calculate)
        button_layout.addWidget(self.button_calculate)

        # Set up keyboard shortcuts
        undo_shortcut = QKeySequence(Qt.CTRL + Qt.Key_Z)
        self.undo_shortcut = QShortcut(undo_shortcut, self)
        self.undo_shortcut.activated.connect(self.undo)

        calculate_shortcut = QKeySequence(Qt.Key_Return)
        self.calculate_shortcut = QShortcut(calculate_shortcut, self)
        self.calculate_shortcut.activated.connect(self.calculate)

        enter_shortcut = QKeySequence(Qt.Key_Enter)
        self.enter_shortcut = QShortcut(enter_shortcut, self)
        self.enter_shortcut.activated.connect(self.calculate)

        # Custom font for drawing answers
        self.custom_font = QFont("Noteworthy", 30)  # Adjust size as needed

        # Initialize OpenAI client
        self.client = llm  # Assuming llm is properly initialized

    def clear_canvas(self):
        self.canvas.clear_canvas()
        self.pil_image = Image.new("RGB", (1200, 800), (0, 0, 0))
        self.draw_pil = ImageDraw.Draw(self.pil_image)

    def undo(self):
        self.canvas.undo_last_action()
        # Optionally, update the PIL image if needed
        self.pil_image = Image.new("RGB", (1200, 800), (0, 0, 0))
        self.draw_pil = ImageDraw.Draw(self.pil_image)
        for action in self.canvas.actions:
            for start, end in action:
                self.draw_pil.line(
                    [(start.x(), start.y()), (end.x(), end.y())],
                    fill="white",
                    width=5
                )

    def calculate(self):
        base64_image = self.canvas.get_base64_image()

        # Prepare the message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Give the answer to this math equation. Only respond with the answer. Only respond with numbers. NEVER Words. Only answer unanswered expressions. Look for equal sign with nothing on the right of it. If it has an answer already. DO NOT ANSWER it."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]

        llm=llm

        try:
            response = self.client.chat.completions.create(
                # model="gpt-4o",  # Ensure this model name is correct
                messages=messages,
                max_tokens=300,
            )

            answer = response['choices'][0]['message']['content']
            self.draw_answer(answer)

        except Exception as e:
            print(f"Error during calculation: {e}")

    def draw_answer(self, answer):
        # Find the position to draw the answer
        equals_pos = self.canvas.get_last_equals_position()
        x_start = equals_pos.x() + 70
        y_start = equals_pos.y() - 20

        # Draw the text on the QImage for persistence
        painter = QPainter(self.canvas.image)
        painter.setPen(QColor("#FF9500"))
        painter.setFont(self.custom_font)
        painter.drawText(equals_pos, answer)
        painter.end()

        # Update the display
        self.canvas.update()

        # Optionally, also draw on the PIL image
        try:
            pil_font = ImageFont.truetype("arial.ttf", 100)  # Ensure the font file exists
        except IOError:
            pil_font = ImageFont.load_default()

        self.draw_pil.text(
            (x_start, y_start - 50),
            answer,
            font=pil_font,
            fill="#FF9500"
        )

    def keyPressEvent(self, event):
        # Override to handle Return/Enter keys if needed
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.calculate()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrawingApp()
    window.show()
    sys.exit(app.exec_())
