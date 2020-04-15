from PyQt5 import QtCore, Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import *

import time


class ClickLabel(QLabel):
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        QLabel.mousePressEvent(self, event)


class MouseMoveLabel(ClickLabel):

    clicked = QtCore.pyqtSignal(object)
    mouseRelease = QtCore.pyqtSignal(object)
    mouseMove = QtCore.pyqtSignal(object)

    def mouseMoveEvent(self, event):
        self.mouseMove.emit(event)
        QLabel.mouseMoveEvent(self, event)

    def mousePressEvent(self, event):
        self.clicked.emit(event)
        QLabel.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.mouseRelease.emit(event)
        QLabel.mouseReleaseEvent(self, event)


class BackEntry(QLineEdit):
    escPress = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.escPress.emit()
        QLineEdit.keyPressEvent(self, event)


class RewritableLabel(QWidget):
    def __init__(self, id=None, label_text='', variable_text='', enter_command=None, entry_width=50,
                 *args, **kwargs):
        super(RewritableLabel, self).__init__(*args, **kwargs)
        self.__id = id

        self.__name_variable = label_text + ' '
        self.__mark_changed = False

        self.__name_label = QLabel(self.__name_variable)
        self.__name_label.setObjectName('label_name')

        self.__enter_function = enter_command

        self.__variable_entry = BackEntry()
        self.__variable_entry.setFixedWidth(entry_width)
        # self.__variable_entry.installEventFilter(self)
        self.__variable_entry.escPress.connect(self.esc_pressed)
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
        self.setMaximumHeight(50)
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
        self.__name_label.setText(str(self.__name_variable))

    def set_entry_text(self, text=''):
        self.__variable_entry.setText(str(text))

    def set_variable_label(self, value):
        self.__variable_label.setText(str(value))

    def show_variable_label(self):
        self.__variable_entry.hide()
        self.__variable_label.show()

    def get_variable_value(self):
        return self.__variable_label.text()

    def set_mark_changed(self, value):
        self.__mark_changed = value
        if self.__mark_changed:
            self.__name_label.setText("<font color='red'>{}</font>".format(str(self.__name_variable[:-1] + "*")))
        else:
            self.__name_label.setText("<font color='black'>{}</font>".format(str(self.__name_variable)))

    def esc_pressed(self):
        if self.__variable_entry.isVisible():
            self.show_variable_label()


class RemovingCombobox(QWidget):
    def __init__(self, *args, **kwargs):
        super(RemovingCombobox, self).__init__(*args, **kwargs)

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.__default_text = ''
        self.__read_only = ''
        self.__next_special_ID = None

        self.__all_values = {}
        self.__already_selected = []
        self.__ordered_values = []
        self.__backward_values = {}

        self.__combobox = QComboBox(self)
        self.__combobox.setInsertPolicy(QComboBox.NoInsert)
        self.__combobox.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.__combobox.view().setUniformItemSizes(True)
        layout.addWidget(self.__combobox, alignment=QtCore.Qt.AlignBottom)

        self.__add_btn = QPushButton('Add', self)
        self.__add_btn.clicked.connect(self.show_selected)
        layout.addWidget(self.__add_btn, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        self.__command = None

    def test(self):
        print('test')

    def initialize(self, item_list, command=None, button_text: str = 'Add', read_only: bool = True, default_text: str = ''):
        self.__all_values = {}
        self.__already_selected = []
        self.__ordered_values = []
        self.__backward_values = {}
        self.__command = command
        self.__next_special_ID = 1

        self.__combobox.clear()

        self.__add_btn.setText(str(button_text))
        self.__read_only = read_only
        self.__default_text = default_text
        if self.__read_only:
            self.__combobox.setEditable(False)
        else:
            self.__combobox.setEditable(True)

        if self.__read_only and self.__default_text != '':
            self.__ordered_values.append(default_text)
        tmp_set = set()
        prev_set_len = len(tmp_set)

        for i, item_name in enumerate(item_list):
            tmp_set.add(item_name)
            if len(tmp_set) == prev_set_len:
                item_name = self.get_unique_name(item_name)
                tmp_set.add(item_name)
            prev_set_len = len(tmp_set)
            self.__all_values[item_name] = i
            self.__ordered_values.append(item_name)
            self.__backward_values[i] = item_name
        self.update_list()

        self.__add_btn.adjustSize()
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
            self.__command((self.__all_values[selected], selected))

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


class FloatSlider(QSlider):
    onResize = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(FloatSlider, self).__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals
        super().setMinimum(0)
        super().setMaximum(self._max_int)
        self._min_value = 0
        self._max_value = 1

    def value(self):
        return float(super().value()) / self._max_int * (self._max_value - self._min_value) + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / (self._max_value - self._min_value) * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value

    def resizeEvent(self, *args, **kwargs):
        super().resizeEvent(*args, **kwargs)
        self.onResize.emit()

    def wheelEvent(self, *args, **kwargs):
        pass


class DisplaySlider(QWidget):
    def __init__(self, *args, **kwargs):
        super(DisplaySlider, self).__init__(*args, **kwargs)
        self.__id = None
        self.__wrapper_gb = QGroupBox()
        self._slider = FloatSlider(QtCore.Qt.Horizontal)
        self.__entry_label_gb = QFrame(self.__wrapper_gb)
        self.__hide_btn = QPushButton()
        self.__value_label = ClickLabel(self.__entry_label_gb)
        self.__value_entry = BackEntry(self.__entry_label_gb)
        self.initialze_ui()

    def initialze_ui(self):
        v_layout = QVBoxLayout(self)
        v_layout.setSpacing(0)

        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.addWidget(self.__wrapper_gb)
        self.__wrapper_gb.setTitle('Nadpis')
        display_layout = QVBoxLayout()

        display_layout.addWidget(self.__value_label)
        display_layout.addWidget(self.__value_entry)
        self.__entry_label_gb.setLayout(display_layout)

        wrapper_layout = QHBoxLayout()
        self.__wrapper_gb.setLayout(wrapper_layout)
        wrapper_layout.setSpacing(10)
        wrapper_layout.setContentsMargins(10, 15, 5, 5)

        wrapper_layout.addWidget(self._slider)

        self.__hide_btn.setText('X')
        self.__hide_btn.setStyleSheet('QPushButton { color: red; }')
        self.__hide_btn.setMaximumWidth(20)
        wrapper_layout.addWidget(self.__hide_btn, alignment=QtCore.Qt.AlignRight)

        self.__value_label.setText(str(self.get_formated_value()))
        self.__value_label.adjustSize()
        self.__value_label.clicked.connect(self.show_entry)

        self.__value_entry.hide()
        self.__value_entry.setFixedWidth(30)
        self.__value_entry.escPress.connect(self.show_label)
        self.__value_entry.returnPressed.connect(self.return_pressed)

        self._slider.setMaximum(50)
        self._slider.setMinimum(10)
        self._slider.valueChanged.connect(self.on_value_change)
        self._slider.onResize.connect(self.display_value)

    def initialize(self, slider_id, minimum=0, maximum=100, slider_name='Slider', on_change_command=None,
                   hide_command=None, value=None):
        self.__id = slider_id
        self._slider.setMinimum(minimum)
        self._slider.setMaximum(maximum)
        self.__wrapper_gb.setTitle(slider_name)
        if value is not None:
            self._slider.setValue(value)
        if on_change_command is not None:
            self._slider.valueChanged.connect(lambda: on_change_command(self._slider.value()))
        if hide_command is not None:
            self.__hide_btn.clicked.connect(lambda: hide_command(slider_id))

    def on_value_change(self):
        self.display_value()

    def get_formated_value(self):
        new_value = str('{0:.2f}'.format(round(self._slider.value(), 2)))
        return new_value.rstrip('0').rstrip('.') if '.' in new_value else new_value

    def display_value(self):
        self.__value_label.setText(str(self.get_formated_value()))
        self.__value_label.adjustSize()
        self.__entry_label_gb.adjustSize()
        new_x = self.convert_to_pixels(self._slider.value())
        self.__entry_label_gb.move(int(new_x), self._slider.y() - 30)

    def show_entry(self):
        self.__value_label.hide()
        self.__value_entry.setText(str(self.__value_label.text()))
        self.__value_entry.show()
        self.__value_entry.adjustSize()
        self.__value_entry.setFixedHeight(self.__value_label.height())
        self.__entry_label_gb.adjustSize()
        self.display_value()

    def show_label(self):
        self.__value_entry.hide()
        self.__value_label.show()
        self.display_value()

    def convert_to_pixels(self, value):
        hodnota = ((value - self._slider.minimum()) / (self._slider.maximum() - self._slider.minimum()))
        return hodnota * (self._slider.width() - self.__entry_label_gb.width() / 2)

    def validate_entry(self, ):
        try:
            if not (self._slider.minimum() <= float(self.__value_entry.text()) <= self._slider.maximum()):
                return False
            self._slider.setValue(float(self.__value_entry.text()))
            return True
        except ValueError:
            return False

    def esc_pressed(self):
        if self.__variable_entry.isVisible():
            self.show_variable_label()

    def return_pressed(self):
        if self.validate_entry():
            self.show_label()
        else:
            self.__value_entry.setText('err')

    def clear(self):
        self.__hide_btn.deleteLater()
        self.__hide_btn.clicked.disconnect()
        self.__hide_btn = None
        self._slider.deleteLater()
        self._slider.valueChanged.disconnect()
        self._slider = None
        self.__wrapper_gb.deleteLater()
        self.__wrapper_gb = None
        self.deleteLater()


class VariableDisplaySlider(DisplaySlider):
    def __init__(self, *args, **kwargs):
        super(VariableDisplaySlider, self).__init__(*args, **kwargs)
        self.__variable_list = None
        self.__index = None

    def initialize(self, slider_id, minimum=0, maximum=100, slider_name='Slider', on_change_command=None,
                  hide_command=None, variable_list=None, index=None):
        value = None
        if variable_list is not None and index is not None:
            self.set_variable(variable_list, index)
            value = variable_list[index]
        super().initialize(slider_id, minimum, maximum, slider_name, on_change_command, hide_command, value)

    def set_variable(self, var_list, index):
        self.__variable_list = var_list
        self.__index = index
        self._slider.setValue(self.__variable_list[self.__index])

    def on_value_change(self):
        if self.__variable_list is not None:
            self.__variable_list[self.__index] = self._slider.value()
        super().on_value_change()


class CustomMessageBox(QMessageBox):
    def __init__(self, window_title='', text='', yes_button='Yes', no_buton='No', cancel='Cancel', informative_text='',
                 *args, **kwargs):
        super(CustomMessageBox, self).__init__(*args, **kwargs)
        self.setWindowTitle(window_title)
        self.setText(text)
        self.addButton(yes_button, QMessageBox.YesRole)
        self.addButton(no_buton, QMessageBox.NoRole)
        self.addButton(cancel, QMessageBox.RejectRole)
        self.setInformativeText(informative_text)