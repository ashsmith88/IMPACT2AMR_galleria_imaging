"""
A graphical user interface for the mmhelper module
"""
import sys
import logging
import os
import subprocess
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QFileDialog,
    qApp,
    QMessageBox,
    QGridLayout,
    QLabel,
    QLineEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import analysis_suite.main as analysis

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
plt.switch_backend('qt5agg')


class MainWindow(QMainWindow):  # pylint: disable=too-many-instance-attributes
    """
    Main window, mainly just holds the main widget, and a few
    extras
    """

    def __init__(self, debug=False):
        super().__init__()

        self.init_interface()
        self.currently_selected_folder = None
        self.output_folder = None

    def init_interface(self):
        """Initiates the main widget inside the main window"""
        self.statusBar().showMessage("No Folder Added")
        self.main_wid = MainWidget(self)
        self.setCentralWidget(self.main_wid)
        self.setGeometry(300, 300, 800, 300)
        self.setWindowTitle("Galleria Analysis Suite")
        self.show()

    def findfolder(self):
        """allows a file to be selected using an open window explorer"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        foldername = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if foldername:
            self.statusBar().showMessage("Folder added: " + foldername)
            # self.files.append(fileName)
            # self.files = (fileName if
            #              isinstance(fileName, list) else [fileName])
            self.main_wid.startbutton.setEnabled(True)
            self.main_wid.removeselectedfolder.setEnabled(True)
            self.currently_selected_folder = foldername

    def exit_gui(self):
        """Allows the user to safely close the interface"""
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            qApp.quit()

    def remove_folder(self):
        """Function which allows for a selected file to be removed"""
        self.statusBar().showMessage("Folder removed. Please add a new folder")
        self.currently_selected_folder = None
        self.main_wid.removeselectedfolder.setEnabled(False)
        self.main_wid.startbutton.setEnabled(False)


class MainWidget(QWidget):  # pylint: disable=too-many-instance-attributes
    """
    Main widget to display in the MainWindow
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir_name = []
        self.setAcceptDrops(True)
        self.initiate_widget()

    def initiate_widget(self):  # pylint: disable=too-many-statements
        """initiates main widget"""
        self.added_folder_text = ("Please select your folder")

        self.addfolderbutton = QPushButton("Choose Folder")
        self.addfolderbutton.clicked.connect(self.parent().findfolder)

        self.identifier_label_label = QLabel("Enter your file identifier below")
        self.identifier_label_editor = QLineEdit("300")

        self.exitbutton = QPushButton("Exit")
        self.exitbutton.clicked.connect(self.parent().exit_gui)

        self.startbutton = QPushButton("Start Analysis")
        self.startbutton.clicked.connect(self.start_analysis)
        self.startbutton.setEnabled(False)

        self.seeresults = QPushButton("Open results folder")
        self.seeresults.setEnabled(False)
        self.seeresults.clicked.connect(
            lambda: launch_file_explorer(self.dir_name))

        self.addfolder = QLabel("Search for folder or drag and drop")
        self.welcome = QLabel("Welcome to Galleria Analysis")

        self.removeselectedfolder = QPushButton("Remove selected folder")
        self.removeselectedfolder.setEnabled(False)
        self.removeselectedfolder.clicked.connect(self.parent().remove_folder)

        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        self.welcome.setFont(font)

        # edit layout

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.welcome, 0, 0, 1, 3)
        grid.addWidget(self.addfolder, 1, 0)
        grid.addWidget(self.addfolderbutton, 2, 0)
        grid.addWidget(self.identifier_label_label, 3, 0)
        grid.addWidget(self.identifier_label_editor, 4, 0)
        grid.addWidget(self.removeselectedfolder, 5, 0)
        grid.addWidget(self.startbutton, 6, 0)
        grid.addWidget(self.seeresults, 7, 0)
        grid.addWidget(self.exitbutton, 8, 0)

        # set layout
        self.setLayout(grid)

        self.thread = AnalysisThread(self.parent(), self)
        self.thread.finished_analysis.connect(self.update_interface)

    def start_analysis(self):
        """initiates the analysis by running a thread"""
        if self.thread.isRunning():
            self.parent().output_file = None
            # self.setEnabled(True)
            for child in self.children():
                if hasattr(child, "setEnabled"):
                    child.setEnabled(True)
            self.thread.terminate()
            self.startbutton.setText('Start Analysis')
            self.parent().statusBar().showMessage(
                "Run aborted by user. Please add a new file and start again")
            self.output.setText(
                "Optional: Specify output filename. "
                + "N.B. it will be followed by a timestamp")
            self.filesadded.setText(self.added_files_text)
            self.parent().files = []
            self.parent().update_files_added()
            # self.removeselectedfiles.setEnabled(False)
            # self.setAcceptDrops(True)
        else:
            self.startbutton.setText('Stop Analysis')

            self.file_identifier = self.identifier_label_editor.text()
            # self.setEnabled(False)
            for child in self.children():
                if hasattr(child, "setEnabled"):
                    child.setEnabled(False)
            self.startbutton.setEnabled(True)
            self.parent().statusBar().showMessage("Running analysis")
            self.thread.start()

    def update_interface(self, dir_name):
        """updates the interface once the analysis has finished"""
        # self.setEnabled(True)
        for child in self.children():
            if hasattr(child, "setEnabled"):
                child.setEnabled(True)
        self.startbutton.setText('Start Analysis')
        self.addfolderbutton.setEnabled(True)
        if not dir_name:
            self.parent().statusBar().showMessage(
                "Analysis failed :(")
            self.parent.statusBar().showMessage("Folder removed. Please add a new folder")
            self.parent.currently_selected_folder = None
            return

        self.parent().statusBar().showMessage(
            "Analysis finished. Click to see files or add a new folder")
        self.dir_name = dir_name
        self.seeresults.setEnabled(True)

    def dragEnterEvent(self, event):
        # pylint: disable=no-self-use, invalid-name
        """allows files to be drag and dropped on to the interface"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):  # pylint: disable=invalid-name
        """
        Handles items being dropped on the UI
        """
        for url in event.mimeData().urls():
            if os.path.isdir(url.toLocalFile()):
                parent = self.parent()
                path = url.toLocalFile()
                # parent.files = path if isinstance(path, list) else [path,]
                # self.filesadded.setText(path)
                parent.currently_selected_folder = path
                self.startbutton.setEnabled(True)
                parent.statusBar().showMessage("Folder added: " + path)
                self.removeselectedfolder.setEnabled(True)


class AnalysisThread(QThread):  # pylint: disable=too-few-public-methods
    """
    Thread to run the analysis
    """

    finished_analysis = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, parent=None, parent2=None):
        QThread.__init__(self)
        self.parent = parent
        self.mainwidget = parent2

    def run(self):
        """runs the analysis"""
        inputfolder = self.parent.currently_selected_folder
        exposure = self.mainwidget.file_identifier
        dir_name = analysis.run_batch(
            inputfolder,
            exposure=exposure,
            from_gui=True
        )
        self.finished_analysis.emit(dir_name)


def launch_file_explorer(path):
    """
    Try cross-platform file-explorer opening...
    Courtesy of: http://stackoverflow.com/a/1795849/537098
    """
    if sys.platform == 'win32':
        subprocess.Popen(['start', path], shell=True)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        try:
            subprocess.Popen(['xdg-open', path])
        except OSError:
            # er, think of something else to try
            # xdg-open *should* be supported by recent Gnome, KDE, Xfce
            QMessageBox.critical(
                None, "Oops", "\n".join(
                    [
                        "Couldn't launch the file explorer, sorry!"
                        "Manually open %s in your favourite file manager" %
                        path]))


class QLogHandler(logging.Handler):
    """
    Logging handler
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def emit(self, record):
        msg = self.format(record)
        self.parent.log_message.emit(msg)


def run_gui(debug=False):
    """
    Creates a Qt QApplication and creates and runs the MainWindow interface
    """
    app = QApplication(sys.argv)
    ex = MainWindow(debug=debug)
    ex.show()
    app.exec_()

if __name__ == '__main__':
    run_gui()