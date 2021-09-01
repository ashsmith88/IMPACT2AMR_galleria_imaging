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
    QLineEdit,
    QCheckBox,
    QRadioButton,
    QPlainTextEdit,
    QComboBox,
    QListWidget,
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
        self.files = []
        self.fluorescence_stack = None
        self.integrated_fluo = False
        self.output_file = None
        self.currently_selected_file = None
        self.batch = False
        self.debug = debug

    def init_interface(self):
        """Initiates the main widget inside the main window"""
        self.statusBar().showMessage("No File Added")
        self.main_wid = MainWidget(self)
        self.setCentralWidget(self.main_wid)
        self.setGeometry(300, 300, 800, 300)
        self.setWindowTitle("mmhelper")
        self.show()

    def findfile(self):
        """allows a file to be selected using an open window explorer"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select input file", "",
            "All Files (*);;Python Files (*.py)", options=options)
        if filename:
            self.statusBar().showMessage("File added: " + filename)
            # self.files.append(fileName)
            # self.files = (fileName if
            #              isinstance(fileName, list) else [fileName])
            self.main_wid.startbutton.setEnabled(True)
            self.main_wid.outputlabel.setEnabled(True)
            self.main_wid.brightfield_box.setChecked(True)
            self.main_wid.removeselectedfiles.setEnabled(True)
            self.currently_selected_file = filename
            self.update_files_added()

    def exitmmhelper(self):
        """Allows the user to safely close the interface"""
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            qApp.quit()

    def fluoresc_stack(self, state):
        """
        Checks tick box to see if a separate fluorescence stack is included
        """
        self.fluorescence_stack = 1 if state else None

    def integr_fluo(self, state):
        """checks tick box to see if fluorescence is integrated"""
        self.integrated_fluo = state

    # def brightfield_or_fluorescence(self, state):
    #    pass

    def outputcheckbox(self, state):
        """
        Check box which determines if a specific output path has
        been specified
        """
        if state == Qt.Checked:
            self.main_wid.output.setEnabled(True)
        else:
            self.main_wid.output.setEnabled(False)
            self.main_wid.output.setText(
                "Optional: Specify output filename. "
                + "N.B. it will be followed by a timestamp")

    def remove_files(self):
        """Function which allows for a selected file to be removed"""
        self.statusBar().showMessage("File removed. Please add a new file")
        for item in self.main_wid.selectedfiles.selectedItems():
            self.main_wid.selectedfiles.takeItem(
                self.main_wid.selectedfiles.row(item))
            filetoberemoved = item.text()
            self.files.remove(filetoberemoved)
            if not self.files:
                self.main_wid.removeselectedfiles.setEnabled(False)
                self.statusBar().showMessage(
                    "Add a file before starting analysis")
                self.main_wid.startbutton.setEnabled(False)
                self.main_wid.comb_fluorescence.setEnabled(False)
                self.main_wid.comb_fluorescence.setChecked(False)
                self.main_wid.seper_fluorescence.setEnabled(False)
                self.main_wid.seper_fluorescence.setChecked(False)
                self.main_wid.brightfield_box.setChecked(True)

    def manually_entered_file(self):
        """Allows input files to be entered manually"""
        self.currently_selected_file = self.main_wid.filesadded.text()
        self.update_files_added()
        self.main_wid.filesadded.setText(self.main_wid.added_files_text)
        self.main_wid.removeselectedfiles.setEnabled(True)
        self.main_wid.startbutton.setEnabled(True)

    def update_files_added(self):
        """Updates the list of files that has been added"""
        if self.currently_selected_file is not None:
            self.files.append(self.currently_selected_file)
        self.main_wid.selectedfiles.clear()
        for file in self.files:
            self.main_wid.selectedfiles.addItem(str(file))
        self.currently_selected_file = None

    def batch_or_not(self, state):
        """
        Determines the state of the batch tick box
        """
        self.batch = state == Qt.Checked


class MainWidget(QWidget):  # pylint: disable=too-many-instance-attributes
    """
    Main widget to display in the MainWindow
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir_name = []
        self.info_level = "INFO"
        self.setAcceptDrops(True)
        # Unfortunately doesn't seem to be a good way to do this
        # without accessing logging modules protected attribute
        # pylint: disable=protected-access
        self.infolevel_dict = logging._levelToName
        self.initiate_widget()

    def initiate_widget(self):  # pylint: disable=too-many-statements
        """initiates main widget"""
        self.added_files_text = ("write the full file path, manually choose "
                                 + "or drag & drop the files")

        self.addfilesbutton = QPushButton("Choose Files")
        self.addfilesbutton.clicked.connect(self.parent().findfile)

        self.selectedfiles = QListWidget(self)
        # self.selectedfiles.setReadOnly(True)

        self.exitbutton = QPushButton("Exit")
        self.exitbutton.clicked.connect(self.parent().exitmmhelper)

        self.startbutton = QPushButton("Start Analysis")
        self.startbutton.clicked.connect(self.start_analysis)
        self.startbutton.setEnabled(False)

        self.seeresults = QPushButton("Open results folder")
        self.seeresults.setEnabled(False)
        self.seeresults.clicked.connect(
            lambda: launch_file_explorer(self.dir_name))

        self.output = QLineEdit(
            "Optional: Specify output filename. N.B. it will be "
            + "followed by a timestamp")
        self.output.setEnabled(False)
        self.outputlabel = QCheckBox("Use own name for output File:")
        self.outputlabel.setEnabled(False)
        self.outputlabel.stateChanged.connect(self.parent().outputcheckbox)

        self.addfiles = QLabel("Input file:")
        self.welcome = QLabel("Welcome to mmhelper")
        self.filesadded = QLineEdit(self.added_files_text)
        self.filesadded.returnPressed.connect(
            self.parent().manually_entered_file)

        self.removeselectedfiles = QPushButton("Remove selected files")
        self.removeselectedfiles.setEnabled(False)
        self.removeselectedfiles.clicked.connect(self.parent().remove_files)

        self.analysisoptions = QLabel("Select options")

        ###
        # Start file-mode options (i.e. Brightfield only, combined, seperate)
        ###
        self.brightfield_box = QRadioButton("Brightfield Only")
        self.brightfield_box.setToolTip(
            "<b>Default.</b><br>Images are only brightfield and no "
            + "fluorescence analysis is required.")
        self.brightfield_box.setChecked(True)
        # self.brightfield_box.toggled.connect(self.parent().brightfield_or_fluorescence)

        self.comb_fluorescence = QRadioButton("Combined Fluorescence")
        self.comb_fluorescence.setToolTip(
            "Select if stack contains alternating brightfield and "
            + "fluorescent images")
        self.comb_fluorescence.toggled.connect(self.parent().integr_fluo)
        # self.comb_fluorescence.setEnabled(False)

        self.seper_fluorescence = QRadioButton("Separate Fluorescence")
        self.seper_fluorescence.setToolTip(
            "Select if you have a stack of brightfield images and a"
            + " separate stack of matching fluorescent images")
        self.seper_fluorescence.toggled.connect(self.parent().fluoresc_stack)
        # self.seper_fluorescence.setEnabled(False)
        ###
        # End of file-mode options
        ###

        self.set_debug = QLabel("Select info level")
        self.set_debug.setToolTip(
            "Select the level of logging information to display:"
            + "<br><b>Not Set</b> = all information <br><b>Critical</b>"
            + "= only major errors.<br><br>Default is set to <b>Info</b>.")
        self.debug_info_level = QComboBox(self)
        self.debug_info_level.addItems(self.infolevel_dict.values())
        self.debug_info_level.activated[str].connect(self.set_info_level)

        self.batchcheckbox = QCheckBox("Batch run")
        self.batchcheckbox.setEnabled(True)
        self.batchcheckbox.setChecked(False)
        self.batchcheckbox.stateChanged.connect(self.parent().batch_or_not)

        self.logwidget = QPlainTextEdit(self)
        self.logwidget.setReadOnly(True)

        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        self.welcome.setFont(font)

        # edit layout

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.welcome, 0, 0)
        grid.addWidget(self.addfilesbutton, 1, 5)
        grid.addWidget(self.selectedfiles, 2, 1)
        grid.addWidget(self.removeselectedfiles, 2, 5)
        grid.addWidget(self.filesadded, 1, 1)
        grid.addWidget(self.addfiles, 1, 0)
        grid.addWidget(self.exitbutton, 14, 6)
        grid.addWidget(self.outputlabel, 9, 0)
        grid.addWidget(self.output, 9, 1)
        grid.addWidget(self.analysisoptions, 3, 0)
        grid.addWidget(self.brightfield_box, 4, 0)
        grid.addWidget(self.comb_fluorescence, 5, 0)
        grid.addWidget(self.seper_fluorescence, 6, 0)
        grid.addWidget(self.startbutton, 10, 1)
        grid.addWidget(self.seeresults, 11, 1)
        grid.addWidget(self.set_debug, 12, 0)
        grid.addWidget(self.debug_info_level, 12, 1)
        grid.addWidget(self.logwidget, 13, 1)
        grid.addWidget(self.batchcheckbox, 7, 0)

        # set layout
        self.setLayout(grid)

        self.thread = AnalysisThread(self.parent(), self)
        self.thread.finished_analysis.connect(self.update_interface)
        self.thread.log_message.connect(self.add_log_message)

    def set_info_level(self, str_):
        """Sets the information level for the user"""
        self.info_level = str_  # self.infolevel_dict[str]
        return self.info_level
        # print(self.infolevel_dict[str], flush = True)

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

            # self.setEnabled(False)
            for child in self.children():
                if hasattr(child, "setEnabled"):
                    child.setEnabled(False)
            self.startbutton.setEnabled(True)

            self.parent().statusBar().showMessage("Running analysis")
            if self.outputlabel.isChecked():
                self.parent().output_file = str(self.output.text())
            else:
                self.parent().output_file = None
            # self.addfilesbutton.setEnabled(False)
            # self.filesadded.setEnabled(False)
            # self.comb_fluorescence.setEnabled(False)
            # self.seper_fluorescence.setEnabled(False)
            # self.seeresults.setEnabled(False)
            # self.outputlabel.setEnabled(False)
            # self.setAcceptDrops(False)
            self.thread.start()

    def add_log_message(self, msg):
        """takes a message and adds it to the log window"""
        self.logwidget.appendPlainText(msg)

    def update_interface(self, dir_name):
        """updates the interface once the analysis has finished"""
        # self.setEnabled(True)
        for child in self.children():
            if hasattr(child, "setEnabled"):
                child.setEnabled(True)
        self.startbutton.setText('Start Analysis')
        self.addfilesbutton.setEnabled(True)
        if not dir_name:
            self.parent().statusBar().showMessage(
                "Analysis failed :(")
            self.filesadded.setEnabled(True)
            self.filesadded.setText(
                "Type new file path, manually select or drag file")
            return

        self.parent().statusBar().showMessage(
            "Analysis finished. Click to see files or add a new file")
        self.dir_name = dir_name
        self.seeresults.setEnabled(True)
        self.filesadded.setEnabled(True)
        self.filesadded.setText(
            "Type new file path, manually select or drag file")

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
            if os.path.isfile(url.toLocalFile()) or os.path.isdir(
                    url.toLocalFile()):
                parent = self.parent()
                path = url.toLocalFile()
                # parent.files = path if isinstance(path, list) else [path,]
                # self.filesadded.setText(path)
                parent.currently_selected_file = path
                self.startbutton.setEnabled(True)
                self.outputlabel.setEnabled(True)
                parent.statusBar().showMessage("File added: " + path)
                self.removeselectedfiles.setEnabled(True)
                self.brightfield_box.setChecked(True)
                self.parent().update_files_added()


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
        self.loghandler = QLogHandler(self)

    def run(self):
        """runs the analysis"""
        output = self.parent.output_file
        fluo = self.parent.fluorescence_stack
        fluoresc = self.parent.integrated_fluo
        inputfile = self.parent.files
        debug = self.parent.debug
        # This changes the format of the GUI messages
        # self.loghandler.setFormatter(logger.formatter)
        #logger.addHandler(self.loghandler)
        # using the numeric values (20 is default = Info)
        #logger.setLevel(self.mainwidget.info_level)

        if self.parent.batch:
            analysis.batch_run(
                inputfile,
                output=output,
                fluo=fluo,
                fluoresc=fluoresc,
                batch=True,
                debug=debug)
            dir_name = inputfile[0]
        else:
            dir_name = analysis.run_analysis_pipeline(
                inputfile,
                output=output,
                fluo=fluo,
                fluoresc=fluoresc,
                debug=debug,
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