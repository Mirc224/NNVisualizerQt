from PyQt5 import QtGui
from PyQt5 import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import *

class RewritableLabel(QWidget):
    def __init__(self, id=None, label_text='', variable_text='', enter_command=None, entry_width=60, *args, **kwargs):
        super().__init__(*args, **kwargs)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.setFont(font)
        self.__id = id

        self.__name_variable = label_text + ' '
        self.__mark_changed = False

        self.__name_label = QLabel(self.__name_variable)

        self.__enter_function = enter_command

        self.__variable_entry = QLineEdit()
        self.__variable_entry.setFixedWidth(entry_width)
        self.__variable_entry.installEventFilter(self)
        self.__variable_entry.returnPressed.connect(self.enter_pressed)
        self.__variable_entry.hide()
        self.__variable_label = QLabel(variable_text)
        self.__variable_label.mousePressEvent = self.show_entry

        layout = QHBoxLayout()

        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.__name_label, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__variable_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(self.__variable_entry, alignment=QtCore.Qt.AlignLeft)
        self.setFixedHeight(16)
        self.setLayout(layout)

    def enter_pressed(self):
        if self.__enter_function is not None:
            self.__enter_function(self.__id, self.__variable_entry.text())

    def show_entry(self, event):
        self.__variable_entry.setText(self.__variable_label.text())
        self.__variable_label.hide()
        self.__variable_entry.show()

    def set_label_name(self, name):
        self.__name_variable = name + ' '
        self.__name_label.setText(self.__name_variable)

    def set_entry_text(self, text=''):
        self.__variable_entry.setText(text)

    def set_variable_label(self, value):
        self.__variable_label.text(value)

    def show_variable_label(self):
        self.__variable_entry.hide()
        self.__variable_label.show()

    def get_variable_value(self):
        return self.__variable_label.text()

    def set_mark_changed(self, value):
        self.__mark_changed = value
        if self.__mark_changed:
            self.__name_label.text(self.__name_variable[:-1] + "*")
            self.__name_label.styleSheet('color: red')
        else:
            self.__name_label.text(self.__name_variable)
            self.__name_label.styleSheet('color: black')

    def eventFilter(self, source, event):
        if self.__variable_entry.isVisible():
            if event.type() == QtCore.QEvent.KeyPress and source is self.__variable_entry:
                if event.key() == QtCore.Qt.Key_Escape:
                    self.show_variable_label()
                    return True
        return False

class RemovingCombobox(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.__default_text = ''
        self.__read_only = ''
        self.__next_special_ID = None

        self.__all_values = {}
        self.__already_selected = []
        self.__ordered_values = []
        self.__backward_values = {}

        self.__combobox = QComboBox()
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        layout.addWidget(self.__combobox)

        self.__add_btn = QPushButton('Add')
        self.__add_btn.clicked.connect(self.show_selected)
        layout.addWidget(self.__add_btn)

        self.__command = None

    def initialize(self, item_list, command=None, button_text: str = 'Add', read_only: bool = True, default_text: str = ''):
        self.__all_values = {}
        self.__already_selected = []
        self.__ordered_values = []
        self.__backward_values = {}
        self.__command = command
        self.__next_special_ID = 1

        self.__combobox.clear()

        self.__add_btn.setText(button_text)
        self.__read_only = read_only
        self.__default_text = default_text
        if self.__read_only:
            self.__combobox.setEditable(False)
        else:
            self.__combobox.setEditable(True)

        if self.__read_only and self.__default_text != '':
            self.__ordered_values.append(default_text)

        for i, item_name in enumerate(item_list):
            item_name = self.get_unique_name(item_name)
            self.__all_values[item_name] = i
            self.__ordered_values.append(item_name)
            self.__backward_values[i] = item_name

        self.update_list()

        if self.__read_only and self.__default_text != '':
            return self.__ordered_values[1:].copy()
        else:
            return self.__ordered_values.copy()

    def get_unique_name(self, item_name):
        if item_name in self.__all_values:
            new_name = item_name + ' ({})'
            number_of_copy = 1
            while True:
                if new_name.format(number_of_copy) in self.__all_values:
                    number_of_copy += 1
                    continue
                else:
                    item_name = new_name.format(number_of_copy)
                    break
        return item_name

    def get_list_of_visible(self):
        return [layer_name for layer_name in self.__ordered_values if layer_name not in self.__already_selected]

    def show_selected(self):
        selected = self.__combobox.currentText()
        if self.__command is not None and selected in self.__all_values:
            self.__command(self, (self.__all_values[selected], selected))

    def update_list(self):
        visible_list = self.get_list_of_visible()
        self.__combobox.clear()
        self.__combobox.addItems(visible_list)
        if self.__read_only:
            if self.__default_text != '':
                self.__combobox.setCurrentIndex(0)
        else:
            self.__combobox.setCurrentText(self.__default_text)

    def hide_item(self, item_name):
        if item_name in self.__all_values:
            self.__already_selected.append(item_name)
            self.update_list()

    def show_item(self, item_name):
        if item_name in self.__already_selected:
            if self.__read_only:
                if self.__default_text != item_name:
                    self.__already_selected.remove(item_name)
                    self.update_list()
            else:
                self.__already_selected.remove(item_name)
                self.update_list()

    def add_special(self, item_name):
        new_list = []
        starting_index = 0
        if self.__read_only and self.__default_text != '':
            new_list.append(self.__default_text)
            starting_index += 1
        item_name = self.get_unique_name(item_name)
        new_list.append(item_name)
        new_list.extend(self.__ordered_values[starting_index:])
        self.__ordered_values = new_list
        self.__all_values[item_name] = -self.__next_special_ID
        self.__next_special_ID += 1
        self.update_list()
        return item_name

    def clear(self):
        self.__add_btn.destroy()
        self.__combobox.destroy()
        self.__add_btn.deleteLater()
        self.__combobox.deleteLater()
        self.deleteLater()
        self.__already_selected = None
        self.__backward_values = None
        self.__add_btn = None
        self.__all_values = None
        self.__combobox = None
        self.__command = None

    def __del__(self):
        print('mazanie combobox')