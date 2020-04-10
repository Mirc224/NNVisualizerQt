import gc
import matplotlib
import math
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore
from QtGraphicalComponents import *
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib import backend_bases
from matplotlib.figure import Figure
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import proj3d


class PlotingFrame(QWidget):
    def __init__(self, *args, **kwargs):
        '''
        Popis
        --------
        Obsahuje v sebe graf z knižnice matplotlib. Zobrazuje ako vyzerjú transformované body v danej vrstve. Zobrazuje
        aj mriežku.

        Atribúty
        --------
        :var self.__cords = obsahuje odkaz na súradnice bofob, ktoré budú zobrazované
        :var self.__number_of_dim: udáva počet vykresľovaných dimenzií
        :var self.__graph_title: názov, ktorý sa bude zobrazovať vo vykresľovanom grafe
        :var self.__graph_labels: názvy jednotlivých osí
        :var self.__main_graph_frame: odkaz na graph frame, bude použitý na zobrazovanie informácií o rozkliknutom bode
        :var self.__figure: matplotlib figúra
        :var self.__canvas: ide o plátno, na to aby bolo možné použiť matplolib grafy v rámci tkinter
        :var self.__axis: matplotlib osi, získane z figúry
        :var self.__draw_2D: vyjadruje, či sa má graf vykresliť ako 2D
        :var self.__toolbar: matplotlib toolbar na posúvanie približovanie a podobne. Zobrazovaný len pri 2D. Pri 3D
                             je pohľad ovládaný myšou
        :var self.__changed: pre efektívnejší update
        :var self.__ani: animácia pre prekresľovanie grafu pri zmenách. Najjedoduchší spôsob pre interaktívne a
                         dynamické grafy
        '''
        super().__init__(*args, **kwargs)
        self.__cords = np.array([[], [], []])
        self.__initialized = False
        self.__line_cords_tuples = None

        self.__number_of_outputs = None

        self.__draw_polygon = False
        self.__use_colored_labels = False

        self.__number_of_dim = -1
        self.__graph_title = 'Graf'
        self.__graph_labels = ['Label X', 'Label Y', 'Label Z']
        self.__parent_controller = None

        self.__points_config = None
        self.__points_color = None
        self.__used_color = None

        # self.__graph_container = tk.LabelFrame(self.__plot_wrapper_frame.Frame, relief=tk.FLAT)
        # self.__graph_container.pack(fill='both', expand=True)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.__figure = Figure(figsize=(4, 4), dpi=100)
        self.__canvas = FigureCanvasQTAgg(self.__figure)
        # self.__canvas.setStyleSheet("background-color:transparent;")
        layout.addWidget(self.__canvas)
        self.__canvas.mpl_connect('button_press_event', self.on_mouse_double_click)
        self.__axis = self.__figure.add_subplot(111, projection='3d')
        self.__draw_3D = False

        backend_bases.NavigationToolbar2.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to  previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        )

        self.__toolbar = NavigationToolbar(self.__canvas, self)
        self.__toolbar.setMinimumHeight(35)

        layout.addWidget(self.__toolbar)

        self.__changed = False
        self.__change_in_progress = False
        self.__locked_view = False
        # self.__ani = animation.FuncAnimation(self.__figure, self.update_changed, interval=105)

    def initialize(self, controller, number_of_outputs, displayed_cords, points_config: dict,
                   layer_name: str):
        self.__cords = displayed_cords
        self.__parent_controller = controller
        self.__number_of_outputs = number_of_outputs
        self.__change_in_progress = False
        self.__graph_title = layer_name
        self.__points_config = points_config
        self.__points_color = []
        self.__initialized = True
        self.set_graph_dimension(self.__number_of_outputs)

    def set_graph_dimension(self, dimension: int):
        print('zmena dimenzie')
        if dimension >= 3:
            self.__draw_3D = True
        else:
            self.__draw_3D = False

        self.__figure.clf()
        if self.__draw_3D:
            self.__number_of_dim = 3
            self.__axis = self.__figure.add_subplot(111, projection='3d')

            for child in self.__toolbar.children():
                if not isinstance(child, QLayout):
                    if not isinstance(child, QLabel) and not isinstance(child, QWidgetAction):
                        child.setVisible(False)

        else:
            self.__number_of_dim = 2
            self.__axis = self.__figure.add_subplot(111)
            for child in self.__toolbar.children():
                if not isinstance(child, QLayout):
                    if not isinstance(child, QLabel) and not isinstance(child, QWidgetAction):
                        child.setVisible(True)

        self.__changed = True
        self.update_graph()

    def redraw_graph(self):
        print('redraw')
        if self.__locked_view:
            tmpX = self.__axis.get_xlim()
            tmpY = self.__axis.get_ylim()
            if self.__number_of_dim == 3:
                tmpZ = self.__axis.get_zlim()
        self.__axis.clear()
        self.__axis.grid()

        self.__axis.set_title(self.__graph_title)
        self.__axis.set_xlabel(self.__graph_labels[0])
        if self.__number_of_outputs > 1:
            self.__axis.set_ylabel(self.__graph_labels[1])
            if self.__draw_3D:
                self.__axis.set_zlabel(self.__graph_labels[2])

        if self.__cords is not None:
            number_of_cords = len(self.__cords)
            if self.__draw_polygon:
                if self.__number_of_dim == 3:
                    for edge in self.__line_cords_tuples:
                        xs = edge[0][0], edge[1][0]
                        ys = edge[0][1], edge[1][1]
                        zs = edge[0][2], edge[1][2]
                        line = plt3d.art3d.Line3D(xs, ys, zs, color='black', linewidth=1, alpha=0.3)
                        self.__axis.add_line(line)
                if self.__number_of_dim == 2:
                    for edge in self.__line_cords_tuples:
                        xs = edge[0][0], edge[1][0]
                        if number_of_cords == 1:
                            ys = 0, 0
                        else:
                            ys = edge[0][1], edge[1][1]

                        self.__axis.plot(xs, ys, linestyle='-', color='black', linewidth=1, alpha=0.5)

            x_axe_cords = self.__cords[0]
            y_axe_cords = np.zeros_like(self.__cords[0])
            z_axe_cords = np.zeros_like(self.__cords[0])
            self.set_point_color()

            if len(self.__cords[0]) == len(self.__points_color):
                if number_of_cords >= 2:
                    y_axe_cords = self.__cords[1]
                    if number_of_cords > 2:
                        z_axe_cords = self.__cords[2]

                if self.__draw_3D:
                    self.__axis.scatter(x_axe_cords, y_axe_cords, z_axe_cords, c=self.__points_color)
                    for point in self.__points_config['active_labels']:
                        self.__axis.text(x_axe_cords[point],
                                         y_axe_cords[point],
                                         z_axe_cords[point],
                                         self.__points_config['label'][point])
                else:
                    self.__axis.scatter(x_axe_cords, y_axe_cords, c=self.__points_color)
                    for point in self.__points_config['active_labels']:
                        if point < len(self.__points_config['label']):
                            self.__axis.annotate(self.__points_config['label'][point],
                                                (x_axe_cords[point], y_axe_cords[point]))

        if self.__locked_view:
            print('locked view')
            self.__axis.set_xlim(tmpX)
            self.__axis.set_ylim(tmpY)
            if self.__number_of_dim == 3:
                self.__axis.set_zlim(tmpZ)
        self.__canvas.draw()

    def update_graph(self):
        if self.__initialized:
            self.redraw_graph()

    def set_color_label(self, new_value):
        if new_value:
            if self.__points_config['label_color'] is not None:
                self.__use_colored_labels = True
            else:
                self.__use_colored_labels = False
        else:
            self.__use_colored_labels = False

    def set_point_color(self):
        if self.__use_colored_labels:
            if self.__points_config['label_color'] is not None:
                self.__points_color = self.__points_config['label_color'].copy()
            else:
                self.__points_color = self.__points_config['default_color'].copy()
        else:
            self.__points_color = self.__points_config['default_color'].copy()
        for point, color in self.__points_config['different_points_color']:
            self.__points_color[point] = color

    def clear(self):
        self.__figure.clear()
        self.__figure.clf()
        self.__axis.cla()
        self.__canvas.close()
        self.__canvas.deleteLater()
        self.__toolbar.close()
        self.__toolbar.deleteLater()
        self.deleteLater()
        self.__toolbar = None
        self.__figure = None
        self.__canvas = None
        self.__axis = None
        gc.collect()

    def on_mouse_double_click(self, event):
        if event.dblclick and event.button == 1:
            self.__points_config['different_points_color'].clear()
            self.__points_config['active_labels'].clear()

            nearest_x = 3
            nearest_y = 3
            closest_point = -1
            if self.__cords is not None:
                number_of_points = len(self.__cords[0])
                X_point_cords = self.__cords[0]
                if len(self.__cords) > 1:
                    Y_point_cords = self.__cords[1]
                else:
                    Y_point_cords = np.zeros_like(self.__cords[0])

                if self.__draw_3D:
                    if len(self.__cords) == 3:
                        Z_point_cords = self.__cords[2]
                    else:
                        Z_point_cords = np.zeros_like(self.__cords[0])
                # Rychlejsie pocita ale prehadzuje riadky, neviem to vyriesit
                for point in range(number_of_points):
                    if self.__draw_3D:
                        x_2d, y_2d, _ = proj3d.proj_transform(X_point_cords[point], Y_point_cords[point],
                                                              Z_point_cords[point], self.__axis.get_proj())
                        pts_display = self.__axis.transData.transform((x_2d, y_2d))
                    else:
                        pts_display = self.__axis.transData.transform((X_point_cords[point], Y_point_cords[point]))

                    if math.fabs(event.x - pts_display[0]) < 3 and math.fabs(event.y - pts_display[1]) < 3:
                        if nearest_x > math.fabs(event.x - pts_display[0]) and nearest_y > math.fabs(
                                event.y - pts_display[1]):
                            nearest_x = math.fabs(event.x - pts_display[0])
                            nearest_y = math.fabs(event.y - pts_display[1])
                            closest_point = point

                if closest_point != -1:
                    self.__points_config['different_points_color'].append((closest_point, '#F25D27'))
                    if self.__points_config['label'] is not None and \
                            len(self.__points_config['label']) == self.__points_config['number_of_samples']:

                        self.__points_config['active_labels'].append(closest_point)
                self.__parent_controller.require_graphs_redraw()

    def __del__(self):
        print('mazanie graf')

    ############################### GETTERS AND SETTERS ################################
    @property
    def graph_title(self):
        return self.__graph_title

    @graph_title.setter
    def graph_title(self, new_title):
        self.__graph_title = new_title

    @property
    def graph_labels(self):
        return self.__graph_labels

    @graph_labels.setter
    def graph_labels(self, new_labels_list):
        self.__graph_labels = new_labels_list

    @property
    def locked_view(self):
        return self.__locked_view

    @locked_view.setter
    def locked_view(self, value):
        self.__locked_view = value

    @property
    def draw_polygon(self):
        return self.__draw_polygon

    @draw_polygon.setter
    def draw_polygon(self, value):
        self.__draw_polygon = value

    @property
    def is_3d_graph(self):
        return self.__draw_3D

    @is_3d_graph.setter
    def is_3d_graph(self, value):
        if self.__draw_3D != value:
            print('nastavenei na false')
            self.__draw_3D = value
            if self.__draw_3D:
                self.set_graph_dimension(3)
            else:
                self.set_graph_dimension(2)

    @property
    def points_cords(self):
        return self.__cords

    @points_cords.setter
    def points_cords(self, new_cords):
        self.__cords = new_cords

    @property
    def line_tuples(self):
        return self.__line_cords_tuples

    @line_tuples.setter
    def line_tuples(self, new_tuples):
        self.__line_cords_tuples = new_tuples

    @property
    def point_color(self):
        return self.__points_color

    @point_color.setter
    def point_color(self, new_value):
        self.__points_color = new_value

######################### GETTERS and SETTERS ####################
    @property
    def is_initialized(self):
        return self.__initialized

    @is_initialized.setter
    def is_initialized(self, value):
        self.__initialized = value


class LayerWeightControllerFrame(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = None
        self._weights_reference = None
        self._bias_reference = None
        self._active_slider_dict = {}
        self._slider_dict = {}
        self._number_of_sliders = 0

        self._scrollable_window = QScrollArea()
        self._scroll_area_layout = QVBoxLayout()
        self._scroll_area_content = QWidget()
        self._add_slider_rc = RemovingCombobox()
        self.initialize_ui()

    def initialize_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._scrollable_window.setWidgetResizable(True)
        layout.addWidget(self._scrollable_window)

        self._scroll_area_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_area_layout.setSpacing(2)

        self._scroll_area_layout.setAlignment(QtCore.Qt.AlignTop)

        self._scroll_area_content.setLayout(self._scroll_area_layout)

        self._scrollable_window.setWidget(self._scroll_area_content)

        self._scroll_area_layout.addWidget(self._add_slider_rc)

        self._add_slider_rc.setMinimumHeight(100)

    def addSlider_visibility_test(self):
        if len(self._active_slider_dict) == self._possible_number_of_sliders:
            self._add_slider_rc.hide()
        else:
            self._add_slider_rc.show()

    def handle_combobox_input(self, item: tuple):
        if item[0] >= 0:
            if item[1] not in self._active_slider_dict.keys():
                if len(self._active_slider_dict) + 1 > 1500:
                    if self.show_warning() != 0:
                        return
                self.add_slider(item[1])
        else:
            list_of_remaining = self._add_slider_rc.get_list_of_visible()
            # prve dva su default a vsetky, to treba preskocit
            list_of_remaining = list_of_remaining[1:].copy()
            if len(self._active_slider_dict) + len(list_of_remaining) > 1500:
                if self.show_warning() != 0:
                    return
            for name in list_of_remaining:
                self.add_slider(name)

    def show_warning(self):
        warning_dialog = QMessageBox()
        warning_dialog.setIcon(QMessageBox.Warning)
        warning_dialog.setWindowTitle('Warning')
        warning_dialog.setText('You are going to add another slider, this may lead to performance issues.')
        warning_dialog.setInformativeText('Would you like to add slider anyway?')
        warning_dialog.addButton('Yes', QMessageBox.YesRole)
        default = warning_dialog.addButton('No', QMessageBox.NoRole)
        warning_dialog.setDefaultButton(default)
        warning_dialog.exec_()
        return warning_dialog.result()


    def add_slider(self, slider_name):
        pass

    def create_weight_slider(self, slider_name, slider_config):
        pass

    def create_bias_slider(self, slider_config):
        pass

    def remove_slider(self, slider_id: str):
        slider = self._active_slider_dict.pop(slider_id)
        slider.clear()
        self._add_slider_rc.show_item(slider_id)
        self.addSlider_visibility_test()

    def on_slider_change(self, value):
        self._controller.weight_bias_change_signal()

    def clear(self):
        for slider in self._active_slider_dict.values():
            slider.clear()
        self._active_slider_dict = {}
        self._add_slider_rc.clear()
        self._add_slider_rc = None


class NoFMWeightControllerFrame(LayerWeightControllerFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, controller, layer_weights: list, layer_bias: list):
        for weight_slider in self._active_slider_dict.values():
            weight_slider.clear()
        self._controller = controller
        self._possible_number_of_sliders = 0
        self._slider_dict = {}
        self._active_slider_dict = {}
        self._weights_reference = layer_weights
        self._bias_reference = layer_bias

        tmp_ordered_sliders_names = []
        tmp_ordered_sliders_config = []

        if layer_weights is not None:
            for start_neuron in range(len(self._weights_reference)):
                for end_neuron in range(len(self._weights_reference[start_neuron])):
                    layer_name = 'Weight {}-{}'.format(start_neuron, end_neuron)
                    tmp_ordered_sliders_names.append(layer_name)
                    tmp_ordered_sliders_config.append((True, start_neuron, end_neuron))
        if layer_bias is not None:
            for end_neuron in range(len(layer_bias)):
                layer_name = 'Bias {}'.format(end_neuron)
                tmp_ordered_sliders_names.append(layer_name)
                tmp_ordered_sliders_config.append((False, end_neuron))

        self._possible_number_of_sliders = len(tmp_ordered_sliders_names)

        final_name_list = self._add_slider_rc.initialize(tmp_ordered_sliders_names, self.handle_combobox_input,
                                                         'Add weight', False, 'Select weight')

        for i, slider_name in enumerate(final_name_list):
            self._slider_dict[slider_name] = tmp_ordered_sliders_config[i]
        self.addSlider_visibility_test()

        special = 'Show all'
        special = self._add_slider_rc.add_special(special)

    def add_slider(self, slider_name: str):
        slider_config = self._slider_dict[slider_name]
        if slider_config[0]:
            self.create_weight_slider(slider_name, slider_config)
        else:
            self.create_bias_slider(slider_name, slider_config)
        self._add_slider_rc.hide_item(slider_name)

    def create_weight_slider(self, slider_name, slider_config):
        start_neuron = slider_config[1]
        end_neuron = slider_config[2]
        slider_name = 'Weight {}-{}'.format(start_neuron, end_neuron)
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider,
                          self._weights_reference[start_neuron], end_neuron)
        self._scroll_area_layout.insertWidget(self._scroll_area_layout.count() - 1, slider)
        self._active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def create_bias_slider(self, slider_name, slider_config):
        end_neuron = slider_config[1]
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -10, 10, slider_name, self.on_slider_change, self.remove_slider,
                          self._bias_reference, end_neuron)
        self._scroll_area_layout.insertWidget(self._scroll_area_layout.count() - 1, slider)
        self._active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()


class FMWeightControllerFrame(LayerWeightControllerFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, controller, layer_weights: np.array, layer_bias: np.array, feature_map_number):
        for weight_slider in self._active_slider_dict.values():
            weight_slider.clear()

        self._controller = controller
        self._slider_dict = {}
        self._active_slider_dict = {}
        self._weights_reference = layer_weights
        self._bias_reference = layer_bias

        self.initialize_for_fm(feature_map_number)

    def initialize_for_fm(self, feature_map_number):
        for weight_slider in self._active_slider_dict.values():
            weight_slider.clear()

        self._slider_dict = {}
        self._active_slider_dict = {}
        tmp_ordered_sliders_names = []
        tmp_ordered_configuration = []
        if self._weights_reference is not None:
            weight_shape = self._weights_reference.shape[:-1]
            for channel_i in range(weight_shape[2]):
                for row_i in range(weight_shape[0]):
                    for col_i in range(weight_shape[1]):
                        key = f'Channel{channel_i}: Weight {row_i}-{col_i}'
                        tmp_ordered_sliders_names.append(key)
                        tmp_ordered_configuration.append([True, feature_map_number, channel_i, row_i, col_i])
        if self._bias_reference is not None:
            tmp_ordered_sliders_names.append(f'Bias{feature_map_number}')
            tmp_ordered_configuration.append([False, feature_map_number])

        tmp_ordered_sliders_names = self._add_slider_rc.initialize(tmp_ordered_sliders_names, self.handle_combobox_input,
                                                                   'Add weight', False, 'Select weight')

        for i, key in enumerate(tmp_ordered_sliders_names):
            self._slider_dict[key] = tmp_ordered_configuration[i]

        self._possible_number_of_sliders = len(tmp_ordered_sliders_names)

        self.addSlider_visibility_test()
        special = 'Show all'
        special = self._add_slider_rc.add_special(special)

    def add_slider(self, slider_name: str):
        slider_config = self._slider_dict[slider_name]
        if slider_config[0]:
            self.create_weight_slider(slider_name, slider_config)
        else:
            self.create_bias_slider(slider_name, slider_config)
        self._add_slider_rc.hide_item(slider_name)

    def create_weight_slider(self, slider_name, slider_config):
        feature_map = self._weights_reference[:, :, :, slider_config[1]]
        channel = feature_map[:, :, slider_config[2]]
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider,
                          channel[slider_config[3]], slider_config[4])
        self._scroll_area_layout.insertWidget(self._scroll_area_layout.count() - 1, slider)
        self._active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def create_bias_slider(self, slider_name, slider_config):
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -10, 10, slider_name, self.on_slider_change, self.remove_slider,
                          self._bias_reference, slider_config[1])
        self._scroll_area_layout.insertWidget(self._scroll_area_layout.count() - 1, slider)
        self._active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

