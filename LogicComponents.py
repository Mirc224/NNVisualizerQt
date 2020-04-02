import ntpath
import threading
from os import listdir
from os.path import isfile, join, isdir

import cv2
import matplotlib.colors as mcolors
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow import keras

from AdditionalComponents import *

BASIC_POINT_COLOR = '#04B2D9'
from PlotAndControlComponents import *


class GraphLogicLayer:
    def __init__(self, main_graph):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda, ktorá sa stará o logiku podstaty aplikácie, inicializuje základné grafické časti aplikácie, stará sa aj
        o výpočty.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__graph_page:        odkaz na nadradený tkinter widget
        :var self.__mg_frame:  odkaz na hlavné okno, v ktorom sú vykresľované grafy pre jednotlivé vrstvy
        :var self.__input_data:        vstupné dáta, načítané zo súboru
        :var self.__points_config:     informácie o jednotlivých bodoch
        :var self.__polygon_cords:     súradnice vrcholov zobrazovanej mriežky
        :var self.__number_of_layers:  počet vrstiev neurónovej siete
        :var self.__keras_model:       načítaný zo súboru, stará sa o výpočty. Sú v ňom menené váhy. Model so zmenenými
                                       váhami je možné uložiť.
        :var self.__active_layers:     obsahuje poradové čísla jednotlivých vrstiev, neurónovej siete,
                                       ktoré sú zobrazované
        :var self.__neural_layers:     list obsahujúci všetky vrstvy siete podľa ich poradia v rámci štruktúry NN
        :var self.__monitoring_thread: vlákno sledijúce zmenu váh a následne prepočítanie a prekreslenie grafov
        :var self.__is_running:        premmenná pre monitorovacie vlákno, ktorá značí, či ešte program beží
        :var self.__changed_layer_q:   zásobnik s unikátnymi id zmenených vrstiev. ID predstavuje poradové číslo vrstvy
        :var self.__condition_var:     podmienková premenná, signalizujúca zmenu a potrebu preopočítania súradníc

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param parent: odkaz na nadradený tkinter widget, v ktorom majú byť vykreslené jednotlivé komponenty
        """
        # Grafické komponenty
        self.__mg_frame = main_graph

        # Zobrazované dáta
        self.__input_data = None
        self.__points_config = None
        self.__polygon_cords = None

        # Štruktúra siete, jednotlivé vrstvy, aktívne vrstvy pre zrýchlenie výpočtu
        self.__number_of_layers = 0
        self.__keras_model = None
        self.__active_layers = None
        self.__neural_layers = list()
        self.__keras_layers = list()

        # Vlákno sledujúce zmeny, aby sa zlepšil pocit s používania - menej sekajúce ovládanie
        self.__changed_layer_q = QueueSet()
        self.__condition_var = threading.Condition()
        self.__monitoring_thread = threading.Thread(target=self.monitor_change)

        # Spustenie monitorovacieho vlákna.
        self.__is_running = True
        self.__monitoring_thread.setDaemon(True)
        self.__monitoring_thread.start()

    def initialize(self, model: keras.Model):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializuje hodnoty pre triedu GraphLogicLayer a tak isto aj pre triedy MainGraphFrame.
        Vytvorí triedy NeuralLayer pre každú vrstvu v modeli.
        Spustí monitorovacie vlákno.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param model: načítaný keras model
        """
        for layer in self.__neural_layers:
            layer.clear()

        self.__keras_model = model
        self.__number_of_layers = len(self.__keras_model.layers)
        self.__neural_layers = list()
        self.__keras_layers = list()
        self.__active_layers = list()

        self.__polygon_cords = None
        self.__input_data = None

        # Nastavenie inicializácia konfigu pre vstupné body.
        self.__points_config = dict()
        self.reset_points_config()

        # Nastavenie základnej konfiguracie.
        i = 0
        for layer in self.__keras_model.layers:
            self.__keras_layers.append(layer)
            neural_layer = NeuralLayer(self, layer, i)
            neural_layer.initialize(self.__points_config)
            self.__neural_layers.append(neural_layer)
            i += 1

        self.__mg_frame.initialize(self)

    def recalculate(self, starting_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prepočíta súradnice bodov na jednotlivých vrstvách na základe nastavených váh.
        Nie je možné použiť výstup jednej vrstvy ako vstup ďalšej vrstvy pre zrýchlenie výpočtu, aby sa nepočítali viac
        krát veci čo už boli vypočítané. Preto je použité pre čiastočné zrýchlenie výpočtu využité viac vlakien, každé
        pre jednu vrstvu.
        """
        # jedno vlakno 0.7328296000000023 s
        # viac vlakien rychlejsie o 100ms

        # Je možné paralelizovat výpočty. Pre každú rátanú vrstvu je použité jedno vlákno.
        threads = []
        start = time.perf_counter()
        layers_for_update = [layer_number for layer_number in self.__active_layers if layer_number >= starting_layer]
        self.recalculate_cords(layers_for_update)
        layers_for_update = [layer_number for layer_number in layers_for_update
                             if layer_number >= starting_layer and self.__neural_layers[layer_number].calculate_polygon]
        self.recalculate_grid(layers_for_update)
        end = time.perf_counter()
        print(f'Predict Calculation time {end - start} s')

    def recalculate_cords(self, layers_for_update):
        if self.__input_data is not None:
            activations = self.get_activation_for_layer(self.__input_data, layers_for_update)
            if not isinstance(activations, list):
                activations = [activations]
            for i, layer_number in enumerate(layers_for_update):
                self.__neural_layers[layer_number].point_cords = activations[i].transpose()

    def recalculate_grid(self, layers_for_update):
        if self.__polygon_cords is not None:
            activations_start = self.get_activation_for_layer(self.__polygon_cords[0], layers_for_update)
            activations_end = self.get_activation_for_layer(self.__polygon_cords[1], layers_for_update)
            if not isinstance(activations_start, list):
                activations_start = [activations_start]
                activations_end = [activations_end]
            for layer_number in layers_for_update:
                start_points = activations_start[layer_number].transpose()
                end_points = activations_end[layer_number].transpose()
                self.__neural_layers[layer_number].polygon_cords_tuples = [start_points, end_points]

    def monitor_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda je spustená v monitorovaciom vlákne. Vlákno beží počas celého behu programu.
        """
        # Sleduje zmeny váh a biasov na vrstvách.
        while self.__is_running:
            self.__condition_var.acquire()

            # Čaká, kým sa neobjaví zmena a potom otestuje, či je v zásobníku nejaká vrstva.
            while not self.__changed_layer_q.is_empty():
                self.__condition_var.wait()
                # Otestuje či náhodou beh programu už neskončil.
                if not self.__is_running:
                    self.__condition_var.release()
                    return
            if not self.__is_running:
                self.__condition_var.release()
                return
            # Vytvorenie kópie zásobníka. Vyčistenie zásboníka.
            actual_changed = self.__changed_layer_q.copy()
            self.__changed_layer_q.clear()
            self.__condition_var.release()

            # Aplikovanie zmien na zmeneých vrstvách. Nájdenie vrstvy, od ktorej je potrebné aplikovať zmeny.
            starting_layer_number = self.__number_of_layers
            for layer_number in actual_changed:
                if layer_number < starting_layer_number:
                    starting_layer_number = layer_number
                layer = self.__neural_layers[layer_number]
                self.set_layer_weights_and_biases(layer_number, layer.layer_weights, layer.layer_biases)
            self.recalculate(starting_layer_number)
            self.broadcast_changes(starting_layer_number)

    def get_activation_for_layer(self, input_points, updated_layers_list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti aktiváciu pre danú vrstvu na základe vstupných dát.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param input_points: vstupné body, ktorých aktiváciu chceme získať
        :param layer_number: číslo vrstvy, ktorej výstup chceme získať
        """
        # Aktivacia na jednotlivých vrstvách. Ak je to prvá, vstupná vrstva, potom je aktivácia len vstupné hodnoty.
        calculated_layers = [self.__keras_model.layers[layer_number].output for layer_number in
                             updated_layers_list]
        intermediate_layer_mode = keras.Model(inputs=self.__keras_model.input,
                                              outputs=calculated_layers)
        return intermediate_layer_mode.predict(input_points)

    def set_points_for_layer(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Výpočet súradníc pre vstupy na zadanej vrstve a ich priradenie. Ak je na danej vrstve zvolená aj možnosť
        zobrazenia mriežky, sú aj tieto súradnice prepočítané a priradené.

        Parametre:
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, pre ktorú sa počíta aktivácia
        """
        # nastavenie vstupných bodov
        if self.__input_data is not None:
            self.__neural_layers[layer_number].point_cords = self.get_activation_for_layer(self.__input_data,
                                                                                           [layer_number]).transpose()

        if self.__neural_layers[layer_number].calculate_polygon:
            self.set_polygon_cords(layer_number)

    def set_polygon_cords(self, layer_number):
        # Výpočet aktivácie pre jednotlivé body hrán polygonu.
        if self.__polygon_cords is not None:
            start_points = self.get_activation_for_layer(self.__polygon_cords[0], [layer_number]).transpose()
            end_points = self.get_activation_for_layer(self.__polygon_cords[1], [layer_number]).transpose()
            self.__neural_layers[layer_number].polygon_cords_tuples = [start_points, end_points]

    def broadcast_changes(self, start_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vyžiada aplikovanie zmien a prekreslenie grafu pre aktívne vrstvy, ktoré majú poradove číslo väčšie ako zadaný
        parameter.\

        Paramatre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo vrstvy. Vrstvy s poradovým číslom väčším ako je toto, budú prekreslené.
        :return:
        """
        # Pre aktívne vrstvy, ktoré sú väčšie ako začiatočná vrstva sa aplikujú vykonané zmeny.
        for layer_number in self.__active_layers:
            if layer_number >= start_layer:
                self.__neural_layers[layer_number].apply_changes()
                self.__neural_layers[layer_number].redraw_graph_if_active()

    def redraw_active_graphs(self, start_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Prekreslenie aktívnych vrstiev na základe ich poradového čísla.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo vrstvy. Vrstvy s poradovým číslom väčším ako je toto, budú prekreslené.
        """
        for layer_number in self.__active_layers:
            if layer_number > start_layer:
                self.__neural_layers[layer_number].redraw_graph_if_active()

    def set_layer_weights_and_biases(self, layer_number, layer_weights, layer_biases):
        # Nastvaenie hodnôt a biasu priamo do keras modelu.
        self.__keras_layers[layer_number].set_weights([np.array(layer_weights), np.array(layer_biases)])

    def signal_change_on_layer(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pridá do zásobníku zmien, poradové číslo vrstvy, na ktorej došlo k zmene váh.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: poradové číslo vrstvy, na ktorej došlo k zmene váh
        """
        self.__condition_var.acquire()
        self.__changed_layer_q.add(layer_number)
        self.__condition_var.notify()
        self.__condition_var.release()

    def handle_images_input(self, input_path):
        if self.__keras_model is not None:
            if isinstance(input_path, list):
                images_arr_list = self.load_and_reshape_images(input_path, self.__keras_model.input_shape)
                if len(images_arr_list) != 0:
                    print(np.array(images_arr_list).shape)
                    self.__input_data = np.array(images_arr_list).reshape(-1, *self.__keras_model.input_shape[-3:])
                    self.reset_points_config()

                    basic_color_list = self.__points_config['default_color']

                    for _ in range(len(self.__input_data)):
                        basic_color_list.append(BASIC_POINT_COLOR)
                else:
                    return 'No images loaded!'
            else:
                only_dirs = [file_name for file_name in listdir(input_path) if isdir(join(input_path, file_name))]
                input_data_tmp = []
                labels_tmp = []
                for dir_name in only_dirs:
                    dir_path = join(input_path, dir_name)
                    image_path_list = [join(dir_path, file_name) for file_name in listdir(dir_path)
                                       if isfile(join(dir_path, file_name)) and
                                       join(dir_path, file_name).lower().endswith(('.png', '.jpg', '.jpeg'))]

                    images_arr_list = self.load_and_reshape_images(
                        image_path_list,
                        self.__keras_model.input_shape)
                    labels_tmp += [dir_name for _ in range(len(images_arr_list))]
                    input_data_tmp += images_arr_list

                if len(input_data_tmp) != 0:
                    self.__input_data = np.array(input_data_tmp).reshape(-1, *self.__keras_model.input_shape[-3:])
                    self.reset_points_config()
                    basic_color_list = self.__points_config['default_color']

                    for _ in range(len(self.__input_data)):
                        basic_color_list.append(BASIC_POINT_COLOR)

                    self.__points_config['label'] = labels_tmp
                    self.__points_config['label_color'] = self.get_label_color_list(labels_tmp)
                else:
                    return 'No images loaded!'
            print('idem ratat')
            self.recalculate()
            self.broadcast_changes()
            self.__mg_frame.apply_changes_on_options_frame()
            return None
        else:
            return 'No model loaded!'

    def load_points(self, filepath):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Načítanie bodov zo súboru. Je možné načítať typ .txt a .csv .

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: cesta k súboru obashujúcemu vstupy.
        """
        # Načítanie bodov zo súboru.
        # Načítané súbory môžu byť typu txt alebo csv, na základe toho sa zvolí vetva. Txt súbory by mali byť oddelené
        # medzerou.
        if self.__keras_model is not None:
            file_ext = ntpath.splitext(filepath)[1]
            if file_ext == '.txt':
                data = pd.read_csv(filepath, sep=' ', header=None)
            else:
                data = pd.read_csv(filepath, header=None)

            # Načítanie configu do premenných.
            shape_of_input = self.__keras_model.layers[0].input_shape[1]

            # Testovanie, či sa počet features rovná rozmeru vstupu.
            if len(data.columns) == shape_of_input:
                # Zistujeme, či sú všetky hodnoty číselné, ak nie návrat s chybovou hláškou. Ak áno, sú priradené do
                # premennej
                is_column_numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).to_list()
                if False in is_column_numeric:
                    return 'Data columns contains non numeric values!'
                self.__input_data = data.to_numpy()
                self.reset_points_config()

                points_color = self.__points_config['default_color']
                # Všetkým bodom je nastavená defaultná farba.
                points_color.clear()
                for _ in range(len(self.__input_data)):
                    points_color.append(BASIC_POINT_COLOR)
                    # label_color.append(label_color_dict[label])

                # Ak je počet fetures medzi 1 a 4 je vytvorený polygon (mriežka, ktorú je možné zobraziť)
                if 1 < shape_of_input < 4:
                    # Zistí sa minimalná a maximálna hodnota pre každú súradnicu, aby mriežka nadobúdala len rozmery
                    # bodov
                    minimal_cord = np.min(self.__input_data[:, :shape_of_input], axis=0).tolist()
                    maximal_cord = np.max(self.__input_data[:, :shape_of_input], axis=0).tolist()
                    if shape_of_input == 3:
                        polygon = Polygon(minimal_cord, maximal_cord, [5, 5, 5])
                    elif shape_of_input == 2:
                        polygon = Polygon(minimal_cord, maximal_cord, [5, 5, 5])

                    polygon_peak_cords = np.array(polygon.Peaks)

                    edges_tuples = np.array(polygon.Edges)

                    self.__polygon_cords = []

                    self.__polygon_cords.append(polygon_peak_cords[:, edges_tuples[:, 0]].transpose())

                    self.__polygon_cords.append(polygon_peak_cords[:, edges_tuples[:, 1]].transpose())

                    for layer in self.__neural_layers:
                        layer.possible_polygon = True
                else:
                    for layer in self.__neural_layers:
                        layer.possible_polygon = False

                # Ak prebehlo načítvanaie bez chyby, sú aplikované zmeny,
                self.recalculate()
                self.broadcast_changes()
                self.__mg_frame.apply_changes_on_options_frame()
                return None
            else:
                return 'Different input point dimension!'
        else:
            return 'No Keras model loaded!'

    def load_labels(self, filepath):
        if self.__keras_model is not None:
            file_ext = ntpath.splitext(filepath)[1]
            if file_ext == '.txt':
                data = pd.read_csv(filepath, sep='\n', header=None)
            else:
                data = pd.read_csv(filepath, header=None)

            if self.__input_data is None or len(self.__input_data) == 0:
                return 'No points loaded!'

            if len(data.columns) > 1:
                return 'Invalid number of columns!'

            if len(data[0].values) != len(self.__input_data):
                return 'Invalid number of labels!'

            self.__points_config['label'] = data[0].values.tolist()
            self.__points_config['different_points_color'] = list()
            self.__points_config['active_labels'] = list()

            self.__points_config['label_color'] = self.get_label_color_list(data[0].tolist())

            self.broadcast_changes()
            self.__mg_frame.apply_changes_on_options_frame()

    def get_label_color_list(self, label_list):
        # Z farieb, ktoré sa nachádzajú v premennej matplotlibu sú zvolené základné farby a potom aj ďalšie
        # farby, z ktorých sú zvolené len tmavšie odtiene.
        possible_colors = list(mcolors.BASE_COLORS.keys())
        possible_colors.remove('w')
        for name, value in mcolors.CSS4_COLORS.items():
            if int(value[1:], 16) < 15204888:
                possible_colors.append(name)

        # Zistíme unikátne labels a na základe nich vytvoríme dict, kde je každej label priradená unikátna farba
        # ak je to možné.
        unique_labels = set(label_list)
        label_color_dict = {}
        number_of_unique_colors = len(possible_colors)
        for i, label in enumerate(unique_labels):
            label_color_dict[label] = possible_colors[i % number_of_unique_colors]

        color_label = []
        for label in label_list:
            color_label.append(label_color_dict[label])

        return color_label

    def reset_points_config(self):
        if self.__input_data is not None:
            self.__points_config['number_of_samples'] = len(self.__input_data)
        else:
            self.__points_config['number_of_samples'] = 0
        self.__points_config['default_color'] = list()
        self.__points_config['different_points_color'] = list()
        self.__points_config['active_labels'] = list()
        self.__points_config['label_color'] = None
        self.__points_config['label'] = None

    def load_and_reshape_images(self, img_path_list, input_shape):
        list_of_images_arrays = []
        img_height = input_shape[-3]
        img_width = input_shape[-2]
        flags = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            flags = cv2.IMREAD_GRAYSCALE
        for img_path in img_path_list:
            img_array = cv2.imread(img_path, flags=flags)
            new_array = cv2.resize(img_array, (img_height, img_width))
            print(new_array)
            list_of_images_arrays.append(new_array)
        return list_of_images_arrays

    def __del__(self):
        self.__is_running = False
        self.__condition_var.acquire()
        self.__condition_var.notify()
        self.__condition_var.release()

    ################################# GETTERS and SETTERS ##################################

    @property
    def neural_layers(self):
        return self.__neural_layers

    @neural_layers.setter
    def neural_layers(self, value):
        self.__neural_layers = value

    @property
    def active_layers(self):
        return self.__active_layers

    @active_layers.setter
    def active_layers(self, value):
        self.__active_layers = value


class NeuralLayer:
    def __init__(self, logicLayer: GraphLogicLayer, keras_layer: keras.layers.Layer,
                 layer_number: int, *args, **kwargs):
        '''
        Popis
        --------
        Trieda predstavuje vrstvu neurónovej siete. V rámci nej budú do grafu volené zobrazované súradnice bodov.
        Bude uskutočňovať PCA redukciu priestoru.

        Aribúty
        --------
        :var self.__graph_frame: udržuje v sebe triedu GraphFrame, ktorá vytvá zobrazovací graf a ovládač váh
        :var self.__layer_name: názov vrstvy
        :var self.__layer_number: poradové číslo vrstvy
        :var self.__point_cords: referencia na súradnice bodov v danej vrstve. (hodnoy sa menia v GraphLogicLayer)
        :var self.__displayed_cords: obsahuje súradnice, ktoré budú zobrazené v grafe. Referenciu na tento objekt
                                     obsahuje aj PlotingFrame

        Parametre
        --------
        :param parent: nadradený tkinter Widget
        :param mainGraph: odkaz na MainGraph
        :param logicLayer: odkaz na logicku vrstvu GraphLogicLayer
        :param args:
        :param kwargs:
        '''
        self.__layer_name = keras_layer.get_config()['name']

        self.__has_feature_maps = False
        self.__selected_feature_map = 0
        self.__output_shape = keras_layer.output_shape
        if len(self.__output_shape) > 2:
            self.__has_feature_maps = True
        self.__number_of_outputs = self.__output_shape[-1]
        self.__layer_number = layer_number
        if len(keras_layer.get_weights()) != 0:
            self.__layer_weights = keras_layer.get_weights()[0]
            self.__layer_biases = keras_layer.get_weights()[1]
        else:
            self.__layer_weights = None
            self.__layer_biases = None

        self.__has_points = False

        self.__calculate_polygon = False

        self.__polygon_cords_tuples = None

        self.__layer_config = {}

        self.__points_config = None

        self.__graph_frame = None

        self.__logic_layer = logicLayer
        self.__layer_number = layer_number
        self.__point_cords = np.array([])
        self.__used_cords = []
        self.__neuron_labels = []
        self.__pc_labels = []

        self.__points_method_cords = []

        self.__points_color = None

        self.__visible = False
        self.__weights_changed = False

    def initialize(self, points_config):
        '''
        Parametre
        --------
        :param layer_number: poradové číslo vrstvy
        :param layer_point_cords: refrencia na zoznam súradníc bodov pre danú vrstvu
        :param layer_weights: referencia na hodnoty váh v danej vrstve. Hodnoty sú menené v controllery a používajú sa
                              pri prepočítavaní súradnic v GraphLogicLayer.
        :param layer_bias: referencia na hodnoty bias v danej vrstve. Podobne ako pr layer_weights
        :param layer_name: názov vrstvy, je unikátny pre každú vrstvu, spolu s poradovým číslom sa používa ako ID.
        '''
        # Počet dimenzií, resp. počet súradníc zistíme podľa počtu vnorených listov.

        self.__layer_config = {}
        self.__point_cords = None
        self.__points_config = points_config

        # Počet súradníc ktoré sa majú zobraziť určíme ako menšie z dvojice čísel 3 a počet dimenzií, pretože max počet,
        # ktorý bude možno zobraziť je max 3
        axis_default_names = ['Label X', 'Label Y', 'Label Z']
        self.__neuron_labels = []
        self.__pc_labels = []
        self.__points_method_cords = []
        used_t_sne_components = []
        used_no_method_cords = []
        used_PCA_components = []
        axis_labels = []

        if not self.__has_feature_maps:
            for i in range(self.__number_of_outputs):
                self.__neuron_labels.append(f'Neuron{i}')
                self.__pc_labels.append(f'PC{i + 1}')
        else:
            number_of_rows = self.__output_shape[1]
            number_of_cols = self.__output_shape[2]
            for height_i in range(number_of_rows):
                for width_i in range(number_of_cols):
                    self.__neuron_labels.append(f'FM point {height_i}-{width_i}')
                    self.__pc_labels.append(f'PC{height_i * number_of_cols + width_i + 1}')

        if self.__has_feature_maps:
            output_dimension = self.__output_shape[-3] * self.__output_shape[-2]
        else:
            output_dimension = self.__number_of_outputs

        number_of_cords = min(3, output_dimension)
        if self.__has_feature_maps:
            predefined_cords = [[], []]
            for i in range(number_of_cords):
                x_cord = i
                y_cord = i
                if y_cord > self.__output_shape[-3]:
                    y_cord = self.__output_shape[-3] -1
                if x_cord > self.__output_shape[-2]:
                    x_cord = self.__output_shape[-2] - 1
                predefined_cords[0].append(x_cord)
                predefined_cords[1].append(y_cord)
                axis_labels.append(axis_default_names[i])
            print(predefined_cords)
            used_no_method_cords = predefined_cords
            used_t_sne_components = predefined_cords
            used_PCA_components = predefined_cords
        else:
            for i in range(number_of_cords):
                used_no_method_cords.append(i)
                used_t_sne_components.append(i)
                used_PCA_components.append(i)
                axis_labels.append(axis_default_names[i])

        self.__layer_config['has_feature_maps'] = self.__has_feature_maps
        self.__layer_config['output_shape'] = self.__output_shape
        self.__layer_config['apply_changes'] = False
        self.__layer_config['cords_changed'] = False
        self.__layer_config['has_feature_maps'] = self.__has_feature_maps
        self.__layer_config['output_dimension'] = output_dimension
        self.__layer_config['layer_name'] = self.__layer_name
        self.__layer_config['max_visible_dim'] = number_of_cords

        self.__layer_config['axis_labels'] = axis_labels
        self.__layer_config['number_of_samples'] = 0

        self.__layer_config['possible_polygon'] = False
        self.__layer_config['color_labels'] = False
        self.__layer_config['show_polygon'] = False
        self.__layer_config['locked_view'] = False

        if number_of_cords >= 3:
            self.__layer_config['draw_3d'] = True
        else:
            self.__layer_config['draw_3d'] = False
        self.__layer_config['used_method'] = 'No method'
        self.__layer_config['config_selected_method'] = 'No method'

        #no_method_config = {'displayed_cords': used_no_method_cords}
        print(used_no_method_cords)
        no_method_config = {'displayed_cords': used_no_method_cords}
        pca_config = {'displayed_cords': used_PCA_components,
                      'n_possible_pc': 0,
                      'percentage_variance': None,
                      'largest_influence': None,
                      'options_used_components': used_PCA_components.copy()}

        number_t_sne_components = min(self.__number_of_outputs, 3)
        used_config = {'n_components': number_t_sne_components,
                       'perplexity': 30,
                       'early_exaggeration': 12.0,
                       'learning_rate': 200,
                       'n_iter': 1000}

        parameter_borders = {'n_components': (1, int, number_t_sne_components),
                             'perplexity': (0, float, float("inf")),
                             'early_exaggeration': (0, float, 1000),
                             'learning_rate': (float("-inf"), float, float("inf")),
                             'n_iter': (250, int, float("inf"))
                             }

        t_sne_config = {'used_config': used_config,
                        'options_config': used_config.copy(),
                        'parameter_borders': parameter_borders,
                        'displayed_cords': used_t_sne_components}

        self.__layer_config['no_method_config'] = no_method_config
        self.__layer_config['PCA_config'] = pca_config
        self.__layer_config['t_SNE_config'] = t_sne_config

    def apply_changes(self):
        '''
        Popis
        --------
        Aplikovanie zmien po prepočítaní súradníc.
        '''
        # Je potrbné podľa navolených zobrazovaných súradníc priradiť z prepočítaných jednotlivé súradnice do súradníc
        # zobrazovaných.
        self.set_used_cords()
        if self.__has_points:
            used_method = self.__layer_config['used_method']
            if used_method == 'No method':
                self.apply_no_method()
            elif used_method == 'PCA':
                self.apply_PCA()
            elif used_method == "t-SNE":
                self.apply_t_SNE()
            self.set_points_for_graph()

    def set_points_for_graph(self):
        used_method = self.__layer_config['used_method']
        self.set_displayed_cords()
        if used_method == 'No method':
            self.set_displayed_cords_for_polygon()

    def set_used_cords(self):
        used_method = self.__layer_config['used_method']
        if used_method == 'No method':
            self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        elif used_method == 'PCA':
            self.__used_cords = self.__layer_config['PCA_config']['displayed_cords']
        elif used_method == 't-SNE':
            self.__used_cords = self.__layer_config['t_SNE_config']['displayed_cords']

    def set_displayed_cords_for_polygon(self):
        if self.__polygon_cords_tuples is not None:
            if self.__graph_frame is not None:
                tmp1 = self.__polygon_cords_tuples[0][self.__used_cords].transpose()
                tmp2 = self.__polygon_cords_tuples[1][self.__used_cords].transpose()
                self.__graph_frame.plotting_frame.line_tuples = list(zip(tmp1, tmp2))

    def set_displayed_cords(self):
        if self.__has_feature_maps:
            feature_map_points = feature_map_points = self.__point_cords[self.__selected_feature_map, :, :, :].transpose()
            print(feature_map_points)
            print(feature_map_points[:, self.__used_cords[0], self.__used_cords[1]].transpose())
            self.__graph_frame.plotting_frame.points_cords = feature_map_points[:, self.__used_cords[0], self.__used_cords[1]].transpose()
        else:
            self.__graph_frame.plotting_frame.points_cords = self.__points_method_cords[self.__used_cords]

    def apply_no_method(self):
        # self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        if self.__has_feature_maps:
            # print(self.__point_cords.shape)
            # feature_map_points = self.__point_cords[0, :, :, :].transpose()
            # print(feature_map_points)
            # feature_map_points = self.__point_cords[1, :, :, :].transpose()
            # print(feature_map_points)
            # print(feature_map_points.shape)
            # feature_map_points = self.__point_cords[:, :, :]
            # print(feature_map_points)
            # print(feature_map_points.shape)
            self.__points_method_cords = self.__point_cords.copy()
        else:
            self.__points_method_cords = self.__point_cords.copy()

    def apply_PCA(self):
        pca_config = self.__layer_config['PCA_config']
        # self.__used_cords = pca_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        scaled_data = preprocessing.StandardScaler().fit_transform(points_cords)
        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        pcs_components_transpose = pca_data.transpose()
        self.__points_method_cords = pcs_components_transpose
        number_of_pcs_indexes = min(self.__number_of_outputs, pca.explained_variance_ratio_.size)
        if number_of_pcs_indexes > 0:
            self.__layer_config['PCA_config']['percentage_variance'] = pd.Series(
                np.round(pca.explained_variance_ratio_ * 100, decimals=1),
                index=self.__pc_labels[:number_of_pcs_indexes])
            self.__layer_config['PCA_config']['largest_influence'] = pd.Series(pca.components_[0],
                                                                               index=self.__neuron_labels)

    def apply_t_SNE(self):
        t_sne_config = self.__layer_config['t_SNE_config']
        # self.__used_cords = t_sne_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        number_of_components = t_sne_config['used_config']['n_components']
        tsne = TSNE(**t_sne_config['used_config'])
        transformed_cords = tsne.fit_transform(points_cords).transpose()
        self.__points_method_cords = transformed_cords

    def clear(self):
        '''
        Popis
        --------
        Používat sa pri mazaní. Vyčistí premenné a skryje danú vrstvu.
        '''
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None

    def signal_change(self):
        self.__logic_layer.signal_change_on_layer(self.__layer_number)

    def set_polygon_cords(self):
        self.__logic_layer.set_polygon_cords(self.__layer_number)
        self.apply_changes()

    def require_graphs_redraw(self):
        self.__logic_layer.redraw_active_graphs(-1)

    def redraw_graph_if_active(self):
        if self.__graph_frame is not None:
            self.__graph_frame.redraw_graph()

    def use_config(self):
        print('pouzivam config')
        if self.__visible:
            if self.__layer_config['apply_changes']:
                print('zmenene')
                self.apply_changes()
                self.__layer_config['cords_changed'] = False
                self.__layer_config['apply_changes'] = False
            elif self.__layer_config['cords_changed']:
                self.set_used_cords()
                self.set_points_for_graph()
                self.__layer_config['cords_changed'] = False
            self.__graph_frame.apply_config(self.__layer_config)

    def create_graph_frame(self, options_command, hide_command):
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None
        self.__graph_frame = GraphFrame(self.__has_feature_maps)
        self.__graph_frame.initialize(self, options_command, hide_command)

        self.__visible = True
        return self.__graph_frame

    def __del__(self):
        print('neural layer destroyed')

    ##############################  GETTERS SETTERS ############################################
    @property
    def layer_name(self):
        return self.__layer_name

    @layer_name.setter
    def layer_name(self, name):
        self.__layer_config['layer_name'] = name
        self.__layer_name = name

    @property
    def layer_number(self):
        return self.__layer_number

    @layer_number.setter
    def layer_number(self, new_value):
        self.__layer_number = new_value

    @property
    def config(self):
        return self.__layer_config

    @property
    def points_cords(self):
        return self.__point_cords

    @property
    def points_config(self):
        return self.__points_config

    @points_config.setter
    def points_config(self, value):
        self.__points_config = value

    @points_cords.setter
    def point_cords(self, new_cords):
        self.__point_cords = new_cords
        self.__layer_config['number_of_samples'] = len(new_cords.transpose())
        if self.__point_cords.size == 0:
            self.__has_points = False
        else:
            self.__has_points = True

    @property
    def polygon_cords_tuples(self):
        return self.__polygon_cords_tuples

    @property
    def possible_polygon(self):
        return self.__layer_config['possible_polygon']

    @possible_polygon.setter
    def possible_polygon(self, value):
        self.__layer_config['possible_polygon'] = value

    @polygon_cords_tuples.setter
    def polygon_cords_tuples(self, new_cords_tuples):
        self.__polygon_cords_tuples = new_cords_tuples
        if self.__polygon_cords_tuples is not None:
            self.__layer_config['possible_polygon'] = True
        else:
            self.__layer_config['possible_polygon'] = False
            self.__displayed_lines_cords = None

    @property
    def possible_color_labels(self):
        return self.__points_config['label_color'] is not None

    @property
    def layer_weights(self):
        return self.__layer_weights

    @property
    def layer_biases(self):
        return self.__layer_biases

    @property
    def calculate_polygon(self):
        return self.__calculate_polygon

    @calculate_polygon.setter
    def calculate_polygon(self, value):
        self.__calculate_polygon = value

    @property
    def point_color(self):
        return self.__points_color

    @point_color.setter
    def point_color(self, new_value):
        self.__points_color = new_value

    @property
    def graph_frame(self):
        return self.__graph_frame

    @graph_frame.setter
    def graph_frame(self, value):
        self.__graph_frame = value
        if self.__graph_frame is not None:
            self.__visible = True
        else:
            self.__visible = False

    @property
    def number_of_outputs(self):
        return self.__number_of_outputs

    @property
    def output_shape(self):
        return self.__output_shape

    @property
    def output_dimension(self):
        return self.__layer_config['output_dimension']


class NoFMNeuralLayer:
    def __init__(self, logicLayer: GraphLogicLayer, keras_layer: keras.layers.Layer,
                 layer_number: int, *args, **kwargs):
        '''
        Popis
        --------
        Trieda predstavuje vrstvu neurónovej siete. V rámci nej budú do grafu volené zobrazované súradnice bodov.
        Bude uskutočňovať PCA redukciu priestoru.

        Aribúty
        --------
        :var self.__graph_frame: udržuje v sebe triedu GraphFrame, ktorá vytvá zobrazovací graf a ovládač váh
        :var self.__layer_name: názov vrstvy
        :var self.__layer_number: poradové číslo vrstvy
        :var self.__point_cords: referencia na súradnice bodov v danej vrstve. (hodnoy sa menia v GraphLogicLayer)
        :var self.__displayed_cords: obsahuje súradnice, ktoré budú zobrazené v grafe. Referenciu na tento objekt
                                     obsahuje aj PlotingFrame

        Parametre
        --------
        :param parent: nadradený tkinter Widget
        :param mainGraph: odkaz na MainGraph
        :param logicLayer: odkaz na logicku vrstvu GraphLogicLayer
        :param args:
        :param kwargs:
        '''
        self.__layer_name = keras_layer.get_config()['name']

        self.__has_feature_maps = False
        self.__selected_feature_map = -1
        self.__output_shape = keras_layer.output_shape
        if len(self.__output_shape) > 2:
            self.__has_feature_maps = True

        self.__number_of_outputs = self.__output_shape[-1]
        self.__layer_number = layer_number
        if len(keras_layer.get_weights()) != 0:
            self.__layer_weights = keras_layer.get_weights()[0]
            self.__layer_biases = keras_layer.get_weights()[1]
        else:
            self.__layer_weights = None
            self.__layer_biases = None

        self.__has_points = False

        self.__calculate_polygon = False

        self.__polygon_cords_tuples = None

        self.__layer_config = {}

        self.__points_config = None

        self.__graph_frame = None

        self.__logic_layer = logicLayer
        self.__layer_number = layer_number
        self.__point_cords = np.array([])
        self.__used_cords = []
        self.__neuron_labels = []
        self.__pc_labels = []

        self.__points_method_cords = []

        self.__points_color = None

        self.__visible = False
        self.__weights_changed = False

    def initialize(self, points_config):
        '''
        Parametre
        --------
        :param layer_number: poradové číslo vrstvy
        :param layer_point_cords: refrencia na zoznam súradníc bodov pre danú vrstvu
        :param layer_weights: referencia na hodnoty váh v danej vrstve. Hodnoty sú menené v controllery a používajú sa
                              pri prepočítavaní súradnic v GraphLogicLayer.
        :param layer_bias: referencia na hodnoty bias v danej vrstve. Podobne ako pr layer_weights
        :param layer_name: názov vrstvy, je unikátny pre každú vrstvu, spolu s poradovým číslom sa používa ako ID.
        '''
        # Počet dimenzií, resp. počet súradníc zistíme podľa počtu vnorených listov.

        self.__layer_config = {}
        self.__point_cords = None
        self.__points_config = points_config

        # Počet súradníc ktoré sa majú zobraziť určíme ako menšie z dvojice čísel 3 a počet dimenzií, pretože max počet,
        # ktorý bude možno zobraziť je max 3
        number_of_cords = min(3, self.__number_of_outputs)
        axis_default_names = ['Label X', 'Label Y', 'Label Z']
        self.__neuron_labels = []
        self.__pc_labels = []
        self.__points_method_cords = []
        used_t_sne_components = []
        used_no_method_cords = []
        used_PCA_components = []
        axis_labels = []

        if not self.__has_feature_maps:
            for i in range(self.__number_of_outputs):
                self.__neuron_labels.append(f'Neuron{i}')
                self.__pc_labels.append(f'PC{i + 1}')
        else:
            number_of_rows = self.__output_shape[1]
            number_of_cols = self.__output_shape[2]
            for height_i in range(number_of_rows):
                for width_i in range(number_of_cols):
                    self.__neuron_labels.append(f'FM point {height_i}-{width_i}')
                    self.__pc_labels.append(f'PC{height_i * number_of_cols + width_i + 1}')

        for i in range(number_of_cords):
            used_no_method_cords.append(i)
            used_t_sne_components.append(i)
            used_PCA_components.append(i)
            axis_labels.append(axis_default_names[i])

        self.__layer_config['apply_changes'] = False
        self.__layer_config['cords_changed'] = False
        self.__layer_config['has_feature_maps'] = self.__has_feature_maps
        self.__layer_config['number_of_dimensions'] = self.__number_of_outputs
        self.__layer_config['output_shape'] = self.__number_of_outputs

        self.__layer_config['layer_name'] = self.__layer_name
        self.__layer_config['max_visible_dim'] = number_of_cords

        self.__layer_config['axis_labels'] = axis_labels
        self.__layer_config['number_of_samples'] = 0

        self.__layer_config['possible_polygon'] = False
        self.__layer_config['color_labels'] = False
        self.__layer_config['show_polygon'] = False
        self.__layer_config['locked_view'] = False

        if number_of_cords >= 3:
            self.__layer_config['draw_3d'] = True
        else:
            self.__layer_config['draw_3d'] = False
        self.__layer_config['used_method'] = 'No method'
        self.__layer_config['config_selected_method'] = 'No method'

        no_method_config = {'displayed_cords': used_no_method_cords}
        pca_config = {'displayed_cords': used_PCA_components,
                      'n_possible_pc': 0,
                      'percentage_variance': None,
                      'largest_influence': None,
                      'options_used_components': used_PCA_components.copy()}

        number_t_sne_components = min(self.__number_of_outputs, 3)
        used_config = {'n_components': number_t_sne_components,
                       'perplexity': 30,
                       'early_exaggeration': 12.0,
                       'learning_rate': 200,
                       'n_iter': 1000}

        parameter_borders = {'n_components': (1, int, number_t_sne_components),
                             'perplexity': (0, float, float("inf")),
                             'early_exaggeration': (0, float, 1000),
                             'learning_rate': (float("-inf"), float, float("inf")),
                             'n_iter': (250, int, float("inf"))
                             }

        t_sne_config = {'used_config': used_config,
                        'options_config': used_config.copy(),
                        'parameter_borders': parameter_borders,
                        'displayed_cords': used_t_sne_components}

        self.__layer_config['no_method_config'] = no_method_config
        self.__layer_config['PCA_config'] = pca_config
        self.__layer_config['t_SNE_config'] = t_sne_config

    def apply_changes(self):
        '''
        Popis
        --------
        Aplikovanie zmien po prepočítaní súradníc.
        '''
        # Je potrbné podľa navolených zobrazovaných súradníc priradiť z prepočítaných jednotlivé súradnice do súradníc
        # zobrazovaných.
        self.set_used_cords()
        if self.__has_points:
            used_method = self.__layer_config['used_method']
            if used_method == 'No method':
                self.apply_no_method()
            elif used_method == 'PCA':
                self.apply_PCA()
            elif used_method == "t-SNE":
                self.apply_t_SNE()
            self.set_points_for_graph()

    def set_points_for_graph(self):
        used_method = self.__layer_config['used_method']
        self.set_displayed_cords()
        if used_method == 'No method':
            self.set_displayed_cords_for_polygon()

    def set_used_cords(self):
        used_method = self.__layer_config['used_method']
        if used_method == 'No method':
            self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        elif used_method == 'PCA':
            self.__used_cords = self.__layer_config['PCA_config']['displayed_cords']
        elif used_method == 't-SNE':
            self.__used_cords = self.__layer_config['t_SNE_config']['displayed_cords']

    def set_displayed_cords_for_polygon(self):
        if self.__polygon_cords_tuples is not None:
            if self.__graph_frame is not None:
                tmp1 = self.__polygon_cords_tuples[0][self.__used_cords].transpose()
                tmp2 = self.__polygon_cords_tuples[1][self.__used_cords].transpose()
                self.__graph_frame.plotting_frame.line_tuples = list(zip(tmp1, tmp2))

    def set_displayed_cords(self):
        self.__graph_frame.plotting_frame.points_cords = self.__points_method_cords[self.__used_cords]

    def apply_no_method(self):
        # self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        self.__points_method_cords = self.__point_cords[self.__used_cords]

    def apply_PCA(self):
        pca_config = self.__layer_config['PCA_config']
        # self.__used_cords = pca_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        scaled_data = preprocessing.StandardScaler().fit_transform(points_cords)
        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        pcs_components_transpose = pca_data.transpose()
        self.__points_method_cords = pcs_components_transpose
        number_of_pcs_indexes = min(self.__number_of_outputs, pca.explained_variance_ratio_.size)
        if number_of_pcs_indexes > 0:
            self.__layer_config['PCA_config']['percentage_variance'] = pd.Series(
                np.round(pca.explained_variance_ratio_ * 100, decimals=1),
                index=self.__pc_labels[:number_of_pcs_indexes])
            self.__layer_config['PCA_config']['largest_influence'] = pd.Series(pca.components_[0],
                                                                               index=self.__neuron_labels)

    def apply_t_SNE(self):
        t_sne_config = self.__layer_config['t_SNE_config']
        # self.__used_cords = t_sne_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        number_of_components = t_sne_config['used_config']['n_components']
        tsne = TSNE(**t_sne_config['used_config'])
        transformed_cords = tsne.fit_transform(points_cords).transpose()
        self.__points_method_cords = transformed_cords

    def clear(self):
        '''
        Popis
        --------
        Používat sa pri mazaní. Vyčistí premenné a skryje danú vrstvu.
        '''
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None

    def signal_change(self):
        self.__logic_layer.signal_change_on_layer(self.__layer_number)

    def set_polygon_cords(self):
        self.__logic_layer.set_polygon_cords(self.__layer_number)
        self.apply_changes()

    def require_graphs_redraw(self):
        self.__logic_layer.redraw_active_graphs(-1)

    def redraw_graph_if_active(self):
        if self.__graph_frame is not None:
            self.__graph_frame.redraw_graph()

    def use_config(self):
        print('pouzivam config')
        if self.__visible:
            if self.__layer_config['apply_changes']:
                print('zmenene')
                self.apply_changes()
                self.__layer_config['cords_changed'] = False
                self.__layer_config['apply_changes'] = False
            elif self.__layer_config['cords_changed']:
                self.set_used_cords()
                self.set_points_for_graph()
                self.__layer_config['cords_changed'] = False
            self.__graph_frame.apply_config(self.__layer_config)

    def create_graph_frame(self, options_command, hide_command):
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None
        self.__graph_frame = GraphFrame(self.__has_feature_maps)
        self.__graph_frame.initialize(self, options_command, hide_command)

        self.__visible = True
        return self.__graph_frame

    def __del__(self):
        print('neural layer destroyed')

    ##############################  GETTERS SETTERS ############################################
    @property
    def layer_name(self):
        return self.__layer_name

    @layer_name.setter
    def layer_name(self, name):
        self.__layer_config['layer_name'] = name
        self.__layer_name = name

    @property
    def layer_number(self):
        return self.__layer_number

    @layer_number.setter
    def layer_number(self, new_value):
        self.__layer_number = new_value

    @property
    def config(self):
        return self.__layer_config

    @property
    def points_cords(self):
        return self.__point_cords

    @property
    def points_config(self):
        return self.__points_config

    @points_config.setter
    def points_config(self, value):
        self.__points_config = value

    @points_cords.setter
    def point_cords(self, new_cords):
        self.__point_cords = new_cords
        self.__layer_config['number_of_samples'] = len(new_cords.transpose())
        if self.__point_cords.size == 0:
            self.__has_points = False
        else:
            self.__has_points = True

    @property
    def polygon_cords_tuples(self):
        return self.__polygon_cords_tuples

    @property
    def possible_polygon(self):
        return self.__layer_config['possible_polygon']

    @possible_polygon.setter
    def possible_polygon(self, value):
        self.__layer_config['possible_polygon'] = value

    @polygon_cords_tuples.setter
    def polygon_cords_tuples(self, new_cords_tuples):
        self.__polygon_cords_tuples = new_cords_tuples
        if self.__polygon_cords_tuples is not None:
            self.__layer_config['possible_polygon'] = True
        else:
            self.__layer_config['possible_polygon'] = False
            self.__displayed_lines_cords = None

    @property
    def possible_color_labels(self):
        return self.__points_config['label_color'] is not None

    @property
    def layer_weights(self):
        return self.__layer_weights

    @property
    def layer_biases(self):
        return self.__layer_biases

    @property
    def calculate_polygon(self):
        return self.__calculate_polygon

    @calculate_polygon.setter
    def calculate_polygon(self, value):
        self.__calculate_polygon = value

    @property
    def point_color(self):
        return self.__points_color

    @point_color.setter
    def point_color(self, new_value):
        self.__points_color = new_value

    @property
    def graph_frame(self):
        return self.__graph_frame

    @graph_frame.setter
    def graph_frame(self, value):
        self.__graph_frame = value
        if self.__graph_frame is not None:
            self.__visible = True
        else:
            self.__visible = False

    @property
    def number_of_outputs(self):
        return self.__number_of_outputs

    @property
    def output_shape(self):
        return self.__output_shape


class GraphFrame(QFrame):
    def __init__(self, has_feature_maps, *args, **kwargs):
        '''
        Popis
        --------
        Obaľovacia trieda. Zodpovedá za vytvorenie vykaresľovacieho grafu a ovládača váh.

        Atribúty
        --------
        :var self.__graph: vykresľovacia trieda, zodpoveda za vykresľovanie bodov na vrstve
        :var self.__weight_controller: zmena koeficientov váh v danej vrstve

        Parametre
        --------
        :param neural_layer: odkaz na NeuralLayer, pod ktroú patrí daný GraphFrame
        :param parent: nadradený tkinter Widget
        '''
        super().__init__(*args, **kwargs)
        self.__neural_layer = None
        self.__has_feature_maps = has_feature_maps
        self.__options_btn = QPushButton()
        self.__hide_btn = QPushButton()
        self.__graph = PlotingFrame()

        self.__weight_dict = {}
        self.__weight_names_ordered = []

        if self.__has_feature_maps:
            self.__feature_map_cb = QComboBox()
            self.__weight_controller = FMWeightControllerFrame()
        else:
            self.__feature_map_cb = None
            self.__weight_controller = NoFMWeightControllerFrame()
        self.init_ui()

    def init_ui(self):
        self.setMaximumWidth(500)
        self.setObjectName('graphFrame')
        self.setStyleSheet("#graphFrame { border: 1px solid black; } ")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        buttons_wrapper_layout = QHBoxLayout()
        buttons_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons_wrapper_layout)
        self.__options_btn.setText('Options')
        buttons_wrapper_layout.addWidget(self.__options_btn, alignment=QtCore.Qt.AlignLeft)
        if self.__has_feature_maps:
            buttons_wrapper_layout.addWidget(self.__feature_map_cb)
        self.__hide_btn.setText('Hide')
        buttons_wrapper_layout.addWidget(self.__hide_btn, alignment=QtCore.Qt.AlignRight)
        self.setLayout(layout)
        layout.addWidget(self.__graph)
        layout.addWidget(self.__weight_controller)

    def initialize(self, neural_layer: NeuralLayer, options_command=None, hide_command=None):
        self.__neural_layer = neural_layer
        if options_command is not None:
            self.__options_btn.clicked.connect(lambda: options_command(self.__neural_layer.layer_number))
        if hide_command is not None:
            self.__hide_btn.clicked.connect(lambda: hide_command(self.__neural_layer.layer_number))

        start = time.perf_counter()
        self.__graph.initialize(self, neural_layer.output_dimension, neural_layer.points_cords,
                                neural_layer.points_config, neural_layer.layer_name)

        end = time.perf_counter()
        print(f'Graph initialization {end - start} s')
        if self.__has_feature_maps:
            self.__feature_map_cb.clear()
            output_shape = neural_layer.output_shape
            for feature_map_index in range(output_shape[-1]):
                self.__feature_map_cb.addItem('Feature map {}'.format(feature_map_index))
            self.__feature_map_cb.currentIndexChanged.connect(self.initialize_selected_feature_map)
            self.__weight_controller.initialize(self, neural_layer.layer_weights, neural_layer.layer_biases,
                                                self.__feature_map_cb.currentIndex())
        else:
            print('Weight initilization start')
            time.sleep(2)
            start = time.perf_counter()
            self.__weight_controller.initialize(self, neural_layer.layer_weights, neural_layer.layer_biases)
            end = time.perf_counter()
            print(f'Weight initialization {end - start} s')

    def initialize_selected_feature_map(self):
        if self.__has_feature_maps:
            self.__weight_controller.initialize_for_fm(self.__feature_map_cb.currentIndex())

    def weight_bias_change_signal(self):
        """
        Popis
        --------
        Posúva signál o zmene váhy neurónovej vrstve.
        """
        self.__neural_layer.signal_change()

    def redraw_graph(self):
        self.__graph.update_graph()

    def clear(self):
        self.__graph.clear()
        self.__graph = None
        self.__weight_controller.clear()
        self.__graph = None
        self.deleteLater()
        gc.collect()

    def require_graphs_redraw(self):
        self.__neural_layer.require_graphs_redraw()

    def apply_config(self, config):
        if self.__graph.is_initialized:
            if config['used_method'] == 'No method':
                self.__graph.draw_polygon = config['show_polygon']
            else:
                self.__graph.draw_polygon = False
            self.__graph.locked_view = config['locked_view']
            self.__graph.graph_labels = config['axis_labels']
            self.__graph.is_3d_graph = config['draw_3d']
            self.__graph.set_color_label(config['color_labels'])
            self.redraw_graph()

    def __del__(self):
        print('mazanie graph frame')

    @property
    def plotting_frame(self):
        return self.__graph
