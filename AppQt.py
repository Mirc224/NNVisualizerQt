import gc
import sys
from PyQt5 import QtGui
from PyQt5 import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import ntpath
from tensorflow import keras
from AdditionalComponents import *
import threading
import time
from QtGraphicalComponents import *

import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib import backend_bases
from matplotlib.figure import Figure
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
import numpy as np
import  pandas as pd
import matplotlib.colors as mcolors
BASIC_POINT_COLOUR = '#04B2D9'

class VisualizationApp(QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Spúšťacia trieda. Je obalom celeje aplikácie. Vytvorí programové okno, na celú obrazovku. Táto trieda dedí od
        objektu Tk z build in knižnice tkinter.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.frames: je to dictionary, ktorý obsahuje jednotlivé hlavné stránky aplikácie. Kľúčom k týmto stránkam
                          sú prototypy daných stránok
        """
        # Konštruktor základnej triedy.
        super(QMainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('Bakalarka')
        self.setContentsMargins(0, 0, 0, 0)
        graphPage = GraphPage()
        self.setCentralWidget(graphPage)


class GraphPage(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Podstránka, na ktorej je zobrazovaný rámec s grafmi, váhami a okno s detailmi vrstvy.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__logic_layer: Referencia na logiku aplikácie.
        :var self.__keras_model: Načítaný model, jeho referencia je zaslaná do logic layer, kde sa menia váhy.
                                 Používa sa aj pri ukladaní.
        :var self.__file_path:   Cesta k súboru pre lepší konfort pri načítaní a ukladaní.
        :var self.__file_name:   Meno súboru pre lepší konfort pri ukladaní.
        :var self.__info_label:

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param parent: nadradený tkinter Widget
        """
        super().__init__(*args, **kwargs)
        self.__keras_model = None
        self.__file_path = '.'
        self.__file_name = ''
        self.__logic_layer = None
        self.__button_wrapper = None
        self.__open_load_info_label = None
        self.__save_model_info_label = None
        self.__button_wrapper = None
        self.__main_graph_frame = None

        self.initialize_ui()

        self.__logic_layer = GraphLogicLayer(self.__main_graph_frame)

        self.open_model('./modelik.h5')
        self.load_points('./2d_input.txt')

    def initialize_ui(self):
        vertical_layout = QVBoxLayout()
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        vertical_layout.setSpacing(0)
        self.__button_wrapper = QWidget()
        self.__button_wrapper.setFixedHeight(50)
        vertical_layout.addWidget(self.__button_wrapper)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)

        open_load_layout = QHBoxLayout()
        open_load_layout.setSpacing(5)
        open_load_layout.setContentsMargins(0, 0, 0, 0)

        save_layout = QHBoxLayout()
        save_layout.setSpacing(5)
        save_layout.setContentsMargins(0, 0, 0, 0)

        group_wrapper_open_load = QWidget()

        group_wrapper_save = QWidget()

        self.__open_load_info_label = QLabel('Load keras model.')
        self.__open_load_info_label.setStyleSheet('QLabel { color: orange}')

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.__open_load_info_label.setFont(font)
        self.__save_model_info_label = QLabel('')
        self.__save_model_info_label.setFont(font)

        button_layout.setContentsMargins(0, 0, 0, 0)
        open_btn = QPushButton('Open model')
        open_btn.clicked.connect(self.try_open_model)
        open_load_layout.addWidget(open_btn)
        load_btn = QPushButton('Load points')
        load_btn.clicked.connect(self.try_load_points)
        open_load_layout.addWidget(load_btn)
        open_load_layout.addWidget(self.__open_load_info_label, alignment=QtCore.Qt.AlignRight)
        save_btn = QPushButton('Save model')
        save_btn.clicked.connect(self.save_model)

        save_layout.addWidget(self.__save_model_info_label)

        save_layout.addWidget(save_btn)

        group_wrapper_save.setLayout(save_layout)

        group_wrapper_open_load.setLayout(open_load_layout)

        button_layout.addWidget(group_wrapper_open_load, alignment=QtCore.Qt.AlignLeft)

        button_layout.addWidget(group_wrapper_save, alignment=QtCore.Qt.AlignRight)

        self.__button_wrapper.setLayout(button_layout)

        self.__main_graph_frame = MainGraphFrame()
        vertical_layout.addWidget(self.__main_graph_frame)

        self.setLayout(vertical_layout)

    def try_open_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Načítanie vybraného modelu, pokiaľ bola zvolená nejaká cesta k súboru.
        """
        # self.__open_load_info_label.setText('Try open')
        # self.__open_load_info_label.setStyleSheet("QLabel { color : red; }")
        file_path = QFileDialog.getOpenFileName(self, 'Load keras model', self.__file_path, 'Keras model (*.h5)')[0]
        if file_path != '':
            self.open_model(file_path)

    def open_model(self, filepath):
        """
        Popis:
        ----------------------------------------------------------------------------------------------------------------
        Rozdelenie zadanej cesty k súboru na absolútnu cestu a názov súboru.
        Načítanie súboru na základe cesty.
        Inicializácia logickej vrstvy, načítaným modelom.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: aboslutná cesta k súboru.
        """
        self.__file_path, self.__file_name = ntpath.split(filepath)
        self.__keras_model = keras.models.load_model(filepath)
        self.__logic_layer.initialize(self.__keras_model)
        self.__open_load_info_label.hide()
        self.__save_model_info_label.hide()

    def try_load_points(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Získanie adresy k súboru s hodnotami bodov. Ak nie je načítaný model, zobrazí sa informácia o chybe.
        """
        if self.__keras_model is not None:
            file_path = QFileDialog.getOpenFileName(self, 'Load model input', self.__file_path, 'Text files (*.txt);;CSV files (*.csv)')[0]
            print(file_path)
            if file_path != '':
                self.load_points(file_path)
        else:
            self.__open_load_info_label.setText('You have to load model first!')
            self.__open_load_info_label.setStyleSheet("QLabel { color : red; }")

    def load_points(self, filepath: str):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Načítanie bodov do modelu na základe cesty k súboru so vstupnými dátami. Ak pri načítaní bodov dôjde ku chybe,
        je táto chyba zobrazená.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: cesta k súboru
        :type filepath:  str
        """
        error_message = self.__logic_layer.load_points(filepath)
        if error_message is not None:
            self.__open_load_info_label.setText(error_message)
            self.__open_load_info_label.show()
        else:
            self.__open_load_info_label.hide()

    def save_model(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Uloženie modelu.
        """
        if self.__keras_model is not None:
            file_path = QFileDialog.getSaveFileName(self, 'Save keras model', ntpath.join(self.__file_path,self.__file_name), 'Keras model (*.h5)')[0]
            if file_path != '':
                self.__file_path, self.__file_name = ntpath.split(file_path)
                self.__keras_model.save(file_path)
            self.__save_model_info_label.hide()
        else:
            self.__save_model_info_label.show()
            self.__save_model_info_label.setText('You have to load model first!')
            self.__save_model_info_label.setStyleSheet("QLabel { color : red; }")


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
        :var self.__main_graph_frame:  odkaz na hlavné okno, v ktorom sú vykresľované grafy pre jednotlivé vrstvy
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
        self.__main_graph_frame = main_graph

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
        self.__points_config['label'] = list()
        self.__points_config['default_colour'] = list()
        self.__points_config['different_points_color'] = list()
        self.__points_config['label_colour'] = list()
        self.__points_config['active_labels'] = list()

        # Nastavenie základnej konfiguracie.
        i = 0
        for layer in self.__keras_model.layers:
            self.__keras_layers.append(layer)
            neural_layer = NeuralLayer(self, layer, i)
            neural_layer.initialize(self.__points_config)
            self.__neural_layers.append(neural_layer)
            i += 1

        self.__main_graph_frame.initialize(self)

    def recalculate_cords(self, starting_layer=0):
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
        for layer_number in self.__active_layers:
            if layer_number > starting_layer:
                t = threading.Thread(target=self.set_points_for_layer, args=(layer_number,))
                t.start()
                threads.append(t)
        # Po výpočtoch je potrebné počkať na dokončenie
        for thread in threads:
            thread.join()
        end = time.perf_counter()
        print(f'Calculation time {end - start} s')

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
            self.recalculate_cords(starting_layer_number)
            self.broadcast_changes(starting_layer_number)
            # time.sleep(0.05)

    def get_activation_for_layer(self, input_points, layer_number):
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
        if layer_number == 0:
            return input_points
        else:
            intermediate_layer_mode = keras.Model(inputs=self.__keras_model.input,
                                                  outputs=self.__keras_layers[layer_number - 1].output)
            print(intermediate_layer_mode.predict(input_points))
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
                                                                                           layer_number).transpose()
        if self.__neural_layers[layer_number].calculate_polygon:
            self.set_polygon_cords(layer_number)

    def set_polygon_cords(self, layer_number):
        # Výpočet aktivácie pre jednotlivé body hrán polygonu.
        if self.__polygon_cords is not None:
            start_points = self.get_activation_for_layer(self.__polygon_cords[0], layer_number).transpose()
            end_points = self.get_activation_for_layer(self.__polygon_cords[1], layer_number).transpose()
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
        # Pre aktívne vrstvy, ktoré sú väčšie ako začiatočná vFrstva sa aplikujú vykonané zmeny.
        for layer_number in self.__active_layers:
            if layer_number > start_layer:
                self.__neural_layers[layer_number].apply_changes()
                self.__neural_layers[layer_number].redraw_graph_if_active()
        #self.__main_graph_frame.update_active_options_layer(start_layer)

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
        # layer = self.__neural_layers[layer_number]
        # self.set_layer_weights_and_biases(layer_number, layer.layer_weights, layer.layer_biases)
        # self.recalculate_cords(layer_number)
        # self.broadcast_changes(layer_number)
        # Oznámi, že došlo k zmene na vrstve. Tá je zaradená do zásobníka.
        self.__condition_var.acquire()
        self.__changed_layer_q.add(layer_number)
        self.__condition_var.notify()
        self.__condition_var.release()

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
            points_colour = self.__points_config['default_colour']
            points_label = self.__points_config['label']
            label_colour = self.__points_config['label_colour']

            # Načítanie posledného stĺpca, ktorý by mal obsahovať labels pre jednotlivé vstupy.
            labels_data = data.iloc[:, -1]

            # Testovanie, či je každej hodnote priradený label. Ak nie, návrat s chybovou hláškou.
            if labels_data.isnull().sum() != 0:
                return 'Missing label values!'

            # Hodnoty labels sú zmenené v referencií na labels z posledného stĺpca.
            points_label[:] = labels_data
            # Zvyšné stĺpce okrem posledného sú použité ako vstupné dáta
            data = data.iloc[:, 0:-1]


            # Testovanie, či sa počet features rovná rozmeru vstupu.
            if len(data.columns) == shape_of_input:
                # Zistujeme, či sú všetky hodnoty číselné, ak nie návrat s chybovou hláškou. Ak áno, sú priradené do
                # premennej
                is_column_numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).to_list()
                if False in is_column_numeric:
                    return 'Data columns contains non numeric values!'
                self.__input_data = data.to_numpy()

                # Z farieb, ktoré sa nachádzajú v premennej matplotlibu sú zvolené základné farby a potom aj ďalšie
                # farby, z ktorých sú zvolené len tmavšie odtiene.
                possible_colors = list(mcolors.BASE_COLORS.values())
                for name, value in mcolors.CSS4_COLORS.items():
                    if int(value[1:], 16) < 15204888:
                        possible_colors.append(name)

                # Zistíme unikátne labels a na základe nich vytvoríme dict, kde je každej label priradená unikátna farba
                # ak je to možné.
                unique_labels = labels_data.unique()
                label_colour_dict = {}
                number_of_unique_colors = len(possible_colors)
                for i, label in enumerate(unique_labels):
                    label_colour_dict[label] = possible_colors[i % number_of_unique_colors]

                # Všetkým bodom je nastavená defaultná farba.
                points_colour.clear()
                label_colour.clear()
                for label in points_label:
                    points_colour.append(BASIC_POINT_COLOUR)
                    label_colour.append(label_colour_dict[label])

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
                self.recalculate_cords(-1)
                self.broadcast_changes(-1)
                self.__main_graph_frame.apply_changes_on_options_frame()
                return None
            else:
                return 'Diffrent input point dimension!'
        else:
            return 'No Keras model loaded!'

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


class MainGraphFrame(QWidget):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Hlavné okno podstránky s grafmi. Okno vytvorí scrollovacie okno, do ktorého budú následne vkladané jednotlivé vrstvy
    s grafmi a ich ovládačmi. Vytvorí taktiež options okno, ktoré slúži na zobrazovanie informácii o zvolenem bode a
    možnosti nastavenia zobrazenia grafov v jednotlivých vrstvách.

     Atribúty
     -------------------------------------------------------------------------------------------------------------------
     :var self.__logic_layer:        obsahuje odkaz na vrstvu, ktorá sa zaoberá výpočtami a logickou reprezentáciou
                                     jednotlivých vrstiev a váh.

     :var self.__number_of_layers:   počet vrstiev načítanej neurónovej siete

     :var self.__name_to_order_dict: každej vrstve je postupne prideľované poradové číslo vrstvy, tak ako ide od
                                     začiatku neurónovej siete. Ako kľúč je použitý názov vrstvy.

     :var self.__order_to_name_dict: podľa poradia je každej vrstve pridelené poradové číslo. Poradové číslo je kľúčom
                                     do dict. Jeho hodnotami sú názvy jedntlivých vrstiev.
                                     Ide o spätný dict k __name_to_order_dict.

     :var self.__active_layers:      obsahuje poradové čísla aktívnych vrstiev, pre prídavanie a odoberanie vrstiev z
                                     add_graph_fame

     :var self.__input_panned:       obal pre komponenty InputDataFrame a PannedWindow. Umožňuje podľa potreby
                                     roztiahnuť alebo zúžiť veľkosť input panelu na úkor PannedWindow, ktoré obsahuje
                                     rámce s reprezentáciou jednotlivých vrstiev.

     :var self.__graph_panned:       obal pre ScrollableWindow, ktoré obsahuje rámce v ktorých sú zobrazené jednotlivé
                                     vrstvy.

     :var self.__options_frame:      okno, v ktorom sa zobrazujú možnosti pre zvolenú vrstvu.

     :var self.__scroll_frame:       obsahuje grafické zobrazenie zvolených vrstiev neurónovej siete a
                                     ComboboxAddRemoveFrame, v ktorom sú volené vrstvy, ktoré chceme zobraziť

     :var self.__add_graph_frame:    combobox, z ktorého si vyberáme jednotlivé, ešte neaktívne vrstvy, ktoré tlačidlom
                                     následne zobrazíme. Zobrazuje sa, len ak nie sú zobrazené ešte všetky vrstvy.
     """

    def __init__(self, *args, **kwargs):
        """
        :param parent: nadradený tkinter widget
        :type parent: tk.Widget
        :param logic_layer: odkaz na logickú vrstvu
        :type logic_layer: GraphLogicLayer
        """
        super().__init__(*args, **kwargs)

        #self.__logic_layer = logic_layer
        self.__logic_layer = None
        self.__number_of_layers = 0

        self.__neural_layers = None

        self.__name_to_order_dict = {}

        self.__order_to_name_dict = {}

        self.__active_layers = []

        # Grafické komponenty.
        self.scrollable_frame = QScrollArea()

        self.__scrollbar_content = QWidget()

        self.__graph_area_layout = QHBoxLayout()

        self.__add_remove_layer_rc = RemovingCombobox()

        self.initialize_ui()

        # layer_graph.clear()
        #layer_graph.hide()
        #self.__graph_area_layout.removeWidget(layer_graph)

    def initialize_ui(self):
        horizontal_layout = QHBoxLayout()
        horizontal_layout.setContentsMargins(0, 0, 5, 0)
        horizontal_layout.setSpacing(0)

        options_frame = OptionsFrame()
        options_frame.setMaximumWidth(300)

        self.scrollable_frame.setWidgetResizable(True)
        horizontal_layout.addWidget(options_frame)
        horizontal_layout.addWidget(self.scrollable_frame)

        self.__graph_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__graph_area_layout.setSpacing(0)
        self.__graph_area_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.__scrollbar_content.setLayout(self.__graph_area_layout)

        self.scrollable_frame.setWidget(self.__scrollbar_content)

        self.__add_remove_layer_rc.setFixedWidth(412)

        self.__graph_area_layout.addWidget(self.__add_remove_layer_rc)

        self.setLayout(horizontal_layout)

    def initialize(self, logic_layer: GraphLogicLayer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializačná funkcia. Inicializuje všetky potrbné komponenty tejto vrstvy.
        Vytvorí predbežný zoznam názvov jednotlivých vrstiev. Po inicializácií AddRemoveComboboxFrame budú získane
        unikátne mená, ktoré sa použijú ako kľúče v slovníku.

        Na základe unikátnych mien je vytvorený slovník, ktorý prevádza meno vrstvy na jej poradové číslo a spätný
        slovník, ktorý prevádza poradové číslo na názov vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layers: zoznam NeuralLayer, ktoré bude možné zobraziť.
        :param active_layer:  refenrencia na zoznam aktívnych vrstiev.
        """
        #self.__options_frame.initialize()

        # Vyčistenie slovníkov.
        self.__logic_layer = logic_layer
        self.__order_to_name_dict = {}
        self.__name_to_order_dict = {}

        self.__active_layers = logic_layer.active_layers

        self.__neural_layers = logic_layer.neural_layers
        self.__number_of_layers = len(self.__neural_layers)

        # Vytvorenie predbežného zoznamu názvov vrstiev.
        layer_name_list = []
        for i in range(self.__number_of_layers):
            layer_name_list.append(self.__neural_layers[i].layer_name)

        # Inicializácia AddRemoveComboboxFrame. Funkcia navracia list unikátnych názvov vrstiev.
        # Unikátne názvy su použité ako kľúče v slovníku.
        unique_name_list = self.__add_remove_layer_rc.initialize(layer_name_list, self.show_layer, 'Add layer', True,
                                                                  'Select layer')
        for i, layerName in enumerate(unique_name_list):
            self.__neural_layers[i].layer_name = layerName
            self.__order_to_name_dict[i] = layerName
            self.__name_to_order_dict[layerName] = i

        if len(self.__active_layers) < self.__number_of_layers:
            self.__add_remove_layer_rc.show()
        else:
            self.__add_remove_layer_rc.hide()

        # first_layer_tuple = (self.__neural_layers[0].layer_number, self.__neural_layers[0].layer_name)
        # self.show_layer(first_layer_tuple)

    def show_layer(self, layer_tuple: tuple):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na základe parametra obdržaného z triedy AddRemoveCombobox vytvorí a následne zobrazí zvolenú vrstvu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_tuple:
        (poradové číslo vrstvy ,názov vrstvy) - obashuje hodnotu z triedy AddRemoveCombobox
        """
        layer_name = layer_tuple[1]
        layer_number = self.__name_to_order_dict[layer_name]

        # Otestuje, či je číslo vrstvy valídne.
        if 0 <= layer_number < self.__number_of_layers:
            layer_to_show = None

            # Inicializácia vrstvy na zobrazenie.
            if layer_number < self.__number_of_layers:
                layer_to_show = self.__neural_layers[layer_number]
                graph_frame_widget = GraphFrame()
                layer_to_show.graph_frame = graph_frame_widget
                graph_frame_widget.initialize(layer_to_show, self.show_layer_options_frame, self.hide_layer)
                self.__graph_area_layout.insertWidget(self.__graph_area_layout.count() - 1, graph_frame_widget)


            # Poradové číslo vrstvy je vložené do listu aktívnych vrstiev, ktorý sa využíva pri efektívnejšom updatovaní
            # vykresľovaných grafov.
            self.__active_layers.append(layer_number)
            self.__add_remove_layer_rc.hide_item(layer_name)

            self.__logic_layer.set_points_for_layer(layer_number)
            layer_to_show.apply_changes()
            layer_to_show.redraw_graph_if_active()
            # Ak je počet aktívnych vrstiev rovný celkovému počtu vrstiev je skrytý panel pre pridávanie nových vrstiev.
            if len(self.__active_layers) == self.__number_of_layers:
                self.__add_remove_layer_rc.hide()

    def hide_layer(self, layer_number: int):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje vrstvu, podľa jej poradového čísla.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorá má byť skrytá
        :type layer_number: int
        """
        if layer_number in self.__active_layers:
            layer_name = self.__order_to_name_dict[layer_number]
            layer = self.__neural_layers[layer_number]
            layer.graph_frame.clear()
            layer.graph_frame = None

            self.__active_layers.remove(layer_number)
            self.__add_remove_layer_rc.show_item(layer_name)
            if len(self.__active_layers) < self.__number_of_layers:
                self.__add_remove_layer_rc.show()

    def show_layer_options_frame(self, layer_number):
        if 0 <= layer_number < self.__number_of_layers:
            layer = self.__neural_layers[layer_number]
            self.__options_frame.initialize_with_layer_config(layer, layer.config)

    def apply_changes_on_options_frame(self):
        return
        self.__options_frame.update_selected_config()

    def update_active_options_layer(self, start_layer):
        self.__options_frame.update_active_options_layer(start_layer)


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
        self.__number_of_dimension = keras_layer.input_shape[1]
        self.__layer_number = layer_number

        if len(keras_layer.get_weights()) != 0:
            self.__layer_weights = keras_layer.get_weights()[0]
            self.__layer_biases = keras_layer.get_weights()[1]
        else:
            self.__layer_weights = None
            self.__layer_biases = None

        self.__layer_options_container = None
        self.__options_button = None
        self.__layer_wrapper = None
        self.__graph_frame = None
        self.__hide_button = None
        self.__has_points = False

        self.__computation_in_process = False

        self.__calculate_polygon = False

        self.__polygon_cords_tuples = None

        self.__layer_config = {}

        self.__points_config = None

        self.__logic_layer = logicLayer
        self.__layer_number = layer_number
        self.__point_cords = np.array([])
        self.__used_cords = []
        self.__axis_labels = []
        self.__neuron_labels = []
        self.__pc_labels = []

        self.__points_method_cords = []

        self.__points_colour = None

        self.__points_label = None

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
        self.__computation_in_process = False
        self.__point_cords = np.array([[] for _ in range(self.__number_of_dimension)])
        self.__points_config = points_config

        # Počet súradníc ktoré sa majú zobraziť určíme ako menšie z dvojice čísel 3 a počet dimenzií, pretože max počet,
        # ktorý bude možno zobraziť je max 3
        number_of_cords = min(3, self.__number_of_dimension)
        axis_default_names = ['Label X', 'Label Y', 'Label Z']
        self.__axis_labels = []
        self.__neuron_labels = []
        self.__pc_labels = []
        self.__points_method_cords = []
        used_t_sne_components = []
        used_no_method_cords = []
        used_PCA_components = []

        for i in range(self.__number_of_dimension):
            self.__neuron_labels.append(f'Neuron{i}')
            self.__pc_labels.append(f'PC{i + 1}')

        for i in range(number_of_cords):
            used_no_method_cords.append(i)
            used_t_sne_components.append(i)
            used_PCA_components.append(i)
            self.__axis_labels.append(axis_default_names[i])

        self.__layer_config['apply_changes'] = False
        self.__layer_config['layer_name'] = self.__layer_name
        self.__layer_config['number_of_dimensions'] = self.__number_of_dimension
        self.__layer_config['max_visible_dim'] = number_of_cords
        self.__layer_config['visible_cords'] = self.__used_cords
        self.__layer_config['axis_labels'] = self.__axis_labels
        self.__layer_config['locked_view'] = False
        self.__layer_config['cords_changed'] = False
        self.__layer_config['color_labels'] = False
        self.__layer_config['number_of_samples'] = 0
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

        number_t_sne_components = min(self.__number_of_dimension, 3)
        used_config = {'n_components': number_t_sne_components, 'perplexity': 30, 'early_exaggeration': 12.0,
                       'learning_rate': 200, 'n_iter': 1000}
        parameter_borders = {'n_components': (1, int, number_t_sne_components),
                             'perplexity': (0, float, float("inf")),
                             'early_exaggeration': (0, float, 1000),
                             'learning_rate': (float("-inf"), float, float("inf")),
                             'n_iter': (250, int, float("inf"))
                             }

        t_sne_config = {'used_config': used_config, 'options_config': used_config.copy(),
                        'parameter_borders': parameter_borders,
                        'displayed_cords': used_t_sne_components}
        self.__layer_config['no_method_config'] = no_method_config
        self.__layer_config['PCA_config'] = pca_config
        self.__layer_config['t_SNE_config'] = t_sne_config

        self.__layer_config['possible_polygon'] = False
        self.__layer_config['show_polygon'] = False

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
        #self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        self.__points_method_cords = self.__point_cords[self.__used_cords]

    def apply_PCA(self):
        pca_config = self.__layer_config['PCA_config']
        #self.__used_cords = pca_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        scaled_data = preprocessing.StandardScaler().fit_transform(points_cords)
        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        pcs_components_transpose = pca_data.transpose()
        self.__points_method_cords = pcs_components_transpose
        number_of_pcs_indexes = min(self.__number_of_dimension, pca.explained_variance_ratio_.size)
        if number_of_pcs_indexes > 0:
            self.__layer_config['PCA_config']['percentage_variance'] = pd.Series(
                np.round(pca.explained_variance_ratio_ * 100, decimals=1), index=self.__pc_labels[:number_of_pcs_indexes])
            self.__layer_config['PCA_config']['largest_influence'] = pd.Series(pca.components_[0], index=self.__neuron_labels)

    def apply_t_SNE(self):
        t_sne_config = self.__layer_config['t_SNE_config']
        #self.__used_cords = t_sne_config['displayed_cords']
        points_cords = self.__point_cords.transpose()
        number_of_components = t_sne_config['used_config']['n_components']
        tsne = TSNE(**t_sne_config['used_config'])
        transformed_cords = tsne.fit_transform(points_cords).transpose()
        print(transformed_cords)
        self.__points_method_cords = transformed_cords

    def clear(self):
        '''
        Popis
        --------
        Používat sa pri mazaní. Vyčistí premenné a skryje danú vrstvu.
        '''
        if self.__visible:
            self.__graph_frame.clear()
            self.__layer_options_container.destroy()
            self.__options_button.destroy()
            self.__layer_wrapper.clear()
            self.__hide_button.destroy()
            self.__layer_options_container = None
            self.__options_button = None
            self.__layer_wrapper = None
            self.__hide_button = None
            self.__visible = False

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
        if self.__visible:
            if self.__layer_config['apply_changes']:
                print('zmenene')
                self.apply_changes()
                self.__layer_config['cords_changed'] = False
                self.__layer_config['apply_changes'] = False
            elif self.__layer_config['cords_changed']:
                self.set_used_cords()
                self.set_points_for_graph()
            self.__graph_frame.apply_config(self.__layer_config)

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
    def point_colour(self):
        return self.__points_colour

    @point_colour.setter
    def point_colour(self, new_value):
        self.__points_colour = new_value

    @property
    def graph_frame(self):
        return self.__graph_frame

    @graph_frame.setter
    def graph_frame(self, value):
        self.__graph_frame = value


class OptionsFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """"
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Obsahuje ovladacie prvky pre jednotlivé vrstvy. V rámci nej je možne navoliť zobrazované súradnice pre
        jednotlivé metódy, ofarbiť body na základe ich label, povoliť zobrazenie mriežky, ak je to možné, uzmaknutie
        pohľadu v grafe.
        Je možné aj zvoliť redukciu priestoru a zobraziť požadovaný PCA komponent alebo použiť metódu t-SNE.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__graph_logic:          odkaz na triedu, zaoberajúcu sa logikou aplikácie.
        :var self.__labels_entries_list:  list vstupnov pre zadávanie názvov osi v grafe aktívnej vrstvy.
        :var self.__cords_entries_list:   list vstupov na zadávanie zobrazovaných súradníc
        :var self.__bar_wrapper:          obaľovací widget, stále zobrazený
        :var self.__layer_options_frame:  obaľovací widget, ktorý obaľuje jedntolivé ucelené rámce s možnosťami
        :var self.__layer_name_label:     zobrazuje meno aktívnej vrstvy, pre ktorú sú zobrazované možnosti
        :var self.__cords_choose_frame:   rámec obsahujúci možnosti zobrazovaných súradníc
        :var self.__possible_cords_label: zobrazuje informáciu o rozsahu súradníc, ktoré môžu byť zobrazené.
        :var self.__label_choose_frame:   obaľuje widgety pre vsutpy, ktoré sú použité na načítanie názvu osí grafu
        :var self.__appearance_frame:     obľuje checkboxy, ktoré slúžia na úpravu vzhľadu grafu a jeho správania
                                          pri prekreslení
        :var self.__color_labels:         boolean premenná checboxu, ktorá označuje, či majú byť vstupy s
                                          rovnakým labelom, ofarbené rovnakou farbou
        :var self.__color_labels_check:   checkbox, ktorý zachytáva, či je táto možnosť zvolená alebo nie
        :var self.__lock_view:            boolean premenná checkboxu, ktorá určuje či má graf pri prekresľovaní zmeniť
                                          škálovanie a posun, alebo zachovať naposledy nastavené
        :var self.__lock_view_check:      checkbox, ktorý zachytáva, či je táto možnosť zvolená alebo nie
        :var self.__3d_graph:             boolean premenná checboxu, ktorá označuje či sa má zobrazovať 2D alebo 3D graf
                                          možnosť je dostupná ak je počet neuronov na vrstve väčší alebo rovný ako 3
        :var self.__show_polygon:         boolean premenná checboxu, ktorá označuje či sa má zobrazovať mriežka, v 2D
                                          alebo 3D. Táto možnosť je dostupná ak je vstup do modleu tvoreným 2 alebo 3
                                          vstupmi.
        :var self.__dim_reduction_frame:  obaľovací widget, ktorý drží zoznam možných metód na redukciu priestoru.
                                          Sú tu zobrazované aj informácie z metódy prípadne zoznam nastaviteľných
                                          parametrov, potrebných pre použitie metódy
        :var self.__actual_used_label:    label, ktorý oboznámjue používateľa s práve použitou metódou redukcie
        :var self.__no_method_radio:      radio button, ktorý označuje že je zvolená metóda no_method
        :var self.__PCA_method_radio:     radio button, ktorý označuje že je zvolená metóda PCA
        :var self.__tSNE_method_radio:    radio button, ktorý označuje že je zvolená metóda t-SNE
        :var self.__PCA_info_frame:       obaľuje listboxy obashujúce informácie po použití metódy PCA
        :var self.__PC_explanation_frame: obsahuje listbox, v ktorom je vyjadrená koľko percent variability je
                                          vysvetelných jednotlivými komponentmi PCA
        :var self.__PC_explanation_lb:    listbox v ktorom je zobrazené aká variabilita je vyjadrená jednotlivými PC
        :var self.__PC_scores_frame:      obsahuje listbox, ktorý udáva ktoré neuróny majú najväčšiu váhu pri PCA
        :var self.__PC_scores_lb:         listbox v ktorom sú zoradené neuróny s najäčším vplyvom
        :var self.__tSNE_parameter_frame: obaľovací widget, obsahuje zoznam nastaviteľných parametrov, potrebných pre
                                          metódu t-SNE
        :var self.__tSNE_parameters_dict: slovník, ku v ktorom su k jednotlivým názvom parametrov priradené rewritable
                                          labels
        :var self.__apply_method_btn:     tlačidlo na použitie zvolenej metódy pomocou radio buttons

        Parmetre
        ----------------------------------------------------------------------------------------------------------------
        :param parent: nadradený tkinter Widget
        :param logicalLayer: odkaz na logickú vrstvu grafu
        """
        super().__init__(*args, **kwargs)
        #self.__graph_logic = logicalLayer
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.__labels_entries_list = []
        self.__cords_entries_list = []
        self.__tSNE_parameters_dict = {}
        self.__changed_config = None
        self.__active_layer = None

        # Obalovaci element. Ostáva stále zobrazený.
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(5, 0, 5, 5)
        self.setLayout(main_layout)
        options_groups_layout = QVBoxLayout()
        options_groups_layout.setAlignment(QtCore.Qt.AlignTop)
        options_groups_layout.setSpacing(0)
        options_groups_layout.setContentsMargins(0, 0, 0, 0)

        self.__layer_name_label = QLabel('Layer name')
        self.__layer_name_label.setMargin(10)

        layer_name_font = QtGui.QFont()
        layer_name_font.setPointSize(10)
        layer_name_font.setBold(True)
        self.__layer_name_label.setFont(layer_name_font)

        options_groups_layout.addWidget(self.__layer_name_label, alignment=QtCore.Qt.AlignCenter)

        self.__bar_wrapper = QGroupBox('Layer Options')
        self.__bar_wrapper.setLayout(options_groups_layout)
        main_layout.addWidget(self.__bar_wrapper)

        choose_cords_layout = QVBoxLayout()
        choose_cords_layout.setAlignment(QtCore.Qt.AlignTop)

        self.__choose_cords_gbox = QGroupBox('Choose Cords')
        self.__choose_cords_gbox.setLayout(choose_cords_layout)

        options_groups_layout.addWidget(self.__choose_cords_gbox)

        options_group_title_font = QtGui.QFont()
        options_group_title_font.setPointSize(10)

        self.__possible_cords_label = QLabel('Possible Cords')
        # self.__possible_cords_label.setFont(options_group_title_font)
        self.__possible_cords_label.setContentsMargins(0, 0, 0, 0)

        choose_cords_layout.addWidget(self.__possible_cords_label, alignment=QtCore.Qt.AlignHCenter)

        entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
        for i in range(3):
            rw_label = RewritableLabel(i, entry_names[i], str(0))
            choose_cords_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__cords_entries_list.append(rw_label)

        self.__choose_label_gbox = QGroupBox('Label names')
        options_groups_layout.addWidget(self.__choose_label_gbox)
        choose_labels_layout = QVBoxLayout()
        self.__choose_label_gbox.setLayout(choose_labels_layout)
        choose_labels_layout.setAlignment(QtCore.Qt.AlignTop)

        entry_names = ['X axis label:', 'Y axis label:', 'Z axis label:']
        entry_values = ['Label X', 'Label Y', 'Label Z']
        for i in range(3):
            rw_label = RewritableLabel(i, entry_names[i], entry_values[i])
            choose_labels_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__labels_entries_list.append(rw_label)

        self.__graph_view_gbox = QGroupBox('Graph view options')
        options_groups_layout.addWidget(self.__graph_view_gbox)

        view_options_layout = QVBoxLayout()
        self.__graph_view_gbox.setLayout(view_options_layout)
        view_options_layout.setAlignment(QtCore.Qt.AlignTop)

        self.__graph_view_gbox.setStyleSheet('QCheckBox { font-size: 8pt;};')
        self.__color_labels_cb = QCheckBox('Color labels')
        view_options_layout.addWidget(self.__color_labels_cb)
        self.__lock_view_cb = QCheckBox('Lock view')
        view_options_layout.addWidget(self.__lock_view_cb)
        self.__3d_view_cb = QCheckBox('3D view')
        view_options_layout.addWidget(self.__3d_view_cb)
        self.__show_polygon_cb = QCheckBox('Show polygon')
        view_options_layout.addWidget(self.__show_polygon_cb)


        self.__dim_reduction_gbox = QGroupBox('Dimension reduction')
        options_groups_layout.addWidget(self.__dim_reduction_gbox)
        dim_reduction_layout = QVBoxLayout()
        dim_reduction_layout.setContentsMargins(0, 0, 0, 0)
        dim_reduction_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__dim_reduction_gbox.setLayout(dim_reduction_layout)

        self.__actual_used_reduction_method = QLabel('Actual used: No method')
        self.__actual_used_reduction_method.setMargin(10)
        dim_reduction_layout.addWidget(self.__actual_used_reduction_method, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop )

        radio_button_group = QButtonGroup()
        radio_button_layout = QHBoxLayout()
        radio_button_layout.setContentsMargins(5, 0, 5, 5)
        radio_button_layout.setAlignment(QtCore.Qt.AlignLeft)
        dim_reduction_layout.addLayout(radio_button_layout)

        self.__no_method_rb = QRadioButton('No method')
        self.__no_method_rb.setChecked(True)
        radio_button_group.addButton(self.__no_method_rb)
        radio_button_layout.addWidget(self.__no_method_rb)

        self.__PCA_rb = QRadioButton('PCA')
        radio_button_group.addButton(self.__PCA_rb)
        radio_button_layout.addWidget(self.__PCA_rb)

        self.__t_SNE_rb = QRadioButton('t-SNE')
        radio_button_group.addButton(self.__t_SNE_rb)
        radio_button_layout.addWidget(self.__t_SNE_rb)

        self.__pca_info_gbox = QGroupBox('PCA information')
        self.__pca_info_gbox.setMaximumHeight(200)
        dim_reduction_layout.addWidget(self.__pca_info_gbox, alignment=QtCore.Qt.AlignTop)

        pca_info_layout = QHBoxLayout()
        pca_info_layout.setAlignment(QtCore.Qt.AlignTop)
        pca_info_layout.setContentsMargins(0, 10, 0, 0)
        self.__pca_info_gbox.setLayout(pca_info_layout)

        var_expl_gbox = QGroupBox('PC variance explanation')

        pca_info_layout.addWidget(var_expl_gbox)
        #
        layout_var_expl = QVBoxLayout()
        layout_var_expl.setAlignment(QtCore.Qt.AlignTop)
        self.__var_expl_lb = QListWidget()
        layout_var_expl.setContentsMargins(0, 0, 0, 0)
        var_expl_gbox.setLayout(layout_var_expl)
        layout_var_expl.addWidget(self.__var_expl_lb)

        loading_score_gbox = QGroupBox('Loading scores')
        pca_info_layout.addWidget(loading_score_gbox)

        loading_score_layout = QVBoxLayout()
        loading_score_layout.setAlignment(QtCore.Qt.AlignTop)
        loading_score_gbox.setLayout(loading_score_layout)
        loading_score_layout.setContentsMargins(0, 0, 0, 0)
        self.__load_scores_lb = QListWidget()
        loading_score_layout.addWidget(self.__load_scores_lb)

        self.__t_sne_parameter_gbox = QGroupBox('t-SNE parameters')
        dim_reduction_layout.addWidget(self.__t_sne_parameter_gbox, alignment=QtCore.Qt.AlignTop)
        t_sne_par_layout = QVBoxLayout()
        t_sne_par_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__t_sne_parameter_gbox.setLayout(t_sne_par_layout)

        t_sne_parameter_id_list = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']
        t_sne_parameter_label = ['Number of components:', 'Perplexity:', 'Early exaggeration:', 'Learning rate:',
                                 'Number of iteration:']

        for i in range(len(t_sne_parameter_id_list)):
            t_sne_parameter_rw = RewritableLabel(t_sne_parameter_id_list[i], t_sne_parameter_label[i], '-', None)
            t_sne_par_layout.addWidget(t_sne_parameter_rw, alignment=QtCore.Qt.AlignLeft)
            self.__tSNE_parameters_dict[t_sne_parameter_id_list[i]] = t_sne_parameter_rw


    def initialize(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vyčistenie atribútov a skrytie celého options baru.
        """
        self.__changed_config = None
        self.__active_layer = None
        self.hide_all()

    def initialize_according_to_config(self):
        pass

    def clear_bar(self):
        self.__bar_wrapper.hide()
        self.hide_choose_cords_entries()
        self.hide_label_entries()
        self.hide_graph_view_options()
        self.hide_dimension_reduction_options()

    def hide_choose_cords_entries(self):
        for i in range(3):
            self.__cords_entries_list[i].hide()

    def hide_label_entries(self):
        for i in range(3):
            self.__labels_entries_list[i].hide()

    def hide_graph_view_options(self):
        self.__3d_view_cb.hide()
        self.__show_polygon_cb.hide()

    def hide_dimension_reduction_options(self):
        self.hide_all_methods_information()

    def hide_all_methods_information(self):
        self.__pca_info_gbox.hide()
        self.__t_sne_parameter_gbox.hide()


class GraphFrame(QFrame):
    def __init__(self, *args, **kwargs):
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
        self.__options_btn = QPushButton()
        self.__hide_btn = QPushButton()
        self.__graph = PlotingFrame()
        self.__weight_controller = LayerWeightControllerFrame()

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

        self.__graph.initialize(self, neural_layer.points_cords, neural_layer.points_config, neural_layer.layer_name)
        self.__weight_controller.initialize(self, neural_layer.layer_weights, neural_layer.layer_biases)

    def wight_bias_change_signal(self):
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
        self.deleteLater()

    def require_graphs_redraw(self):
        self.__neural_layer.require_graphs_redraw()

    def __del__(self):
        print('mazanie graph frame')

    @property
    def plotting_frame(self):
        return self.__graph


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
        #self.__plot_wrapper_frame = ResizableWindow(parent, 'bottom', *args, **kwargs)
        #self.__plot_wrapper_frame.pack(fill='both', expand=True)
        self.__cords = [[], [], []]
        self.__line_cords_tuples = None

        self.__draw_polygon = False

        self.__number_of_dim = -1
        self.__graph_title = 'Graf'
        self.__graph_labels = ['Label X', 'Label Y', 'Label Z']
        self.__parent_controller = None

        self.__points_config = None
        self.__points_colour = None
        self.__used_colour = None
        self.__points_label = None
        self.__active_points_label = None
        self.__different_points_colour = None

        # self.__graph_container = tk.LabelFrame(self.__plot_wrapper_frame.Frame, relief=tk.FLAT)
        # self.__graph_container.pack(fill='both', expand=True)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.__figure = Figure(figsize=(4, 4), dpi=100)
        self.__canvas = FigureCanvasQTAgg(self.__figure)
        #self.__canvas.setStyleSheet("background-color:transparent;")
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
        #self.__ani = animation.FuncAnimation(self.__figure, self.update_changed, interval=105)

    def initialize(self, controller: GraphFrame, displayed_cords, points_config: dict, layer_name: str):
        if len(displayed_cords) != 0:
            self.__cords = displayed_cords

        self.__parent_controller = controller
        self.__change_in_progress = False
        self.__graph_title = layer_name
        self.__points_config = points_config
        self.__different_points_colour = self.__points_config['different_points_color']
        self.__used_colour = self.__points_config['default_colour']
        self.__points_colour = []
        self.__points_label = self.__points_config['label']
        self.__active_points_label = self.__points_config['active_labels']
        self.set_graph_dimension(len(displayed_cords))

    def set_graph_dimension(self, dimension: int):
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
        self.__axis.set_title(self.__graph_title)
        self.__axis.set_xlabel(self.__graph_labels[0])
        if len(self.__cords[0]) == len(self.__points_colour):
            if number_of_cords >= 2:
                self.__axis.set_ylabel(self.__graph_labels[1])
                y_axe_cords = self.__cords[1]
                if number_of_cords > 2:
                    z_axe_cords = self.__cords[2]
                    if self.__draw_3D:
                        self.__axis.set_zlabel(self.__graph_labels[2])

            if self.__draw_3D:
                self.__axis.scatter(x_axe_cords, y_axe_cords, z_axe_cords, c=self.__points_colour)
                for point in self.__active_points_label:
                    self.__axis.text(x_axe_cords[point[0]], y_axe_cords[point[0]], z_axe_cords[point[0]], point[1])
            else:
                self.__axis.scatter(x_axe_cords, y_axe_cords, c=self.__points_colour)
                for point in self.__active_points_label:
                    self.__axis.annotate(point[1], (x_axe_cords[point[0]], y_axe_cords[point[0]]))

        if self.__locked_view:
            self.__axis.set_xlim(tmpX)
            self.__axis.set_ylim(tmpY)
            if self.__number_of_dim == 3:
                self.__axis.set_zlim(tmpZ)
        self.__canvas.draw()

    # def update_changed(self, i):
    #     print('update')
    #     if self.__changed:
    #         self.__changed = False
    #         self.__change_in_progress = True
    #         self.redraw_graph()
    #         self.__change_in_progress = False
    #     else:
    #         return
    #         self.__ani.event_source.stop()

    def update_graph(self):
        self.redraw_graph()
        # return
        # self.__changed = True
        # if not self.__change_in_progress:
        #     if self.__ani is not None:
        #         return
        #         self.__ani.event_source.start()

    def set_color_label(self, new_value):
        if new_value:
            self.__used_colour = self.__points_config['label_colour']
        else:
            self.__used_colour = self.__points_config['default_colour']

    def set_point_color(self):
        self.__points_colour = self.__used_colour.copy()
        for point, colour in self.__different_points_colour:
            self.__points_colour[point] = colour

    def clear(self):
        # TODO: Dorobit mazanie, nevycisti sa uplne
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
            self.__different_points_colour.clear()
            self.__active_points_label.clear()

            nearest_x = 3
            nearest_y = 3
            closest_point = -1
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
                self.__different_points_colour.append((closest_point, '#F25D27'))
                if len(self.__points_label) > 0:
                    self.__active_points_label.append((closest_point, self.__points_label[closest_point]))
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
    def point_colour(self):
        return self.__points_colour

    @point_colour.setter
    def point_colour(self, new_value):
        self.__points_colour = new_value


class LayerWeightControllerFrame(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__controller = None
        self.__weights_reference = None
        self.__bias_reference = None
        self.__active_slider_dict = {}
        self.__slider_dict = {}
        self.__possible_number_of_sliders = 0
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.__scrollable_window = QScrollArea()
        self.__scrollable_window.setWidgetResizable(True)
        layout.addWidget(self.__scrollable_window)

        self.__scroll_area_layout = QVBoxLayout()
        self.__scroll_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__scroll_area_layout.setSpacing(2)

        self.__scroll_area_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__scroll_area_content = QWidget()

        self.__scroll_area_content.setLayout(self.__scroll_area_layout)

        self.__scrollable_window.setWidget(self.__scroll_area_content)

        self.__add_slider_rc = RemovingCombobox()
        self.__scroll_area_layout.addWidget(self.__add_slider_rc)

        self.__add_slider_rc.setMinimumHeight(100)
        self.__disable_update = False

    def initialize(self, controller, layer_weights: list, layer_bias: list):
        for weight_slider in self.__active_slider_dict.values():
            weight_slider.deleteLater()

        self.__controller = controller
        self.__disable_update = False
        self.__possible_number_of_sliders = 0
        self.__slider_dict = {}
        self.__active_slider_dict = {}
        self.__weights_reference = layer_weights
        self.__bias_reference = layer_bias

        tmp_ordered_sliders_names = []
        tmp_ordered_sliders_config = []
        if layer_weights is not None:
            for start_neuron in range(len(self.__weights_reference)):
                for end_neuron in range(len(self.__weights_reference[start_neuron])):
                    layer_name = 'Vaha {}-{}'.format(start_neuron, end_neuron)
                    tmp_ordered_sliders_names.append(layer_name)
                    tmp_ordered_sliders_config.append((True, start_neuron, end_neuron))
        if layer_bias is not None:
            for end_neuron in range(len(layer_bias)):
                layer_name = 'Bias {}'.format(end_neuron)
                tmp_ordered_sliders_names.append(layer_name)
                tmp_ordered_sliders_config.append((False, end_neuron))

        self.__possible_number_of_sliders = len(tmp_ordered_sliders_names)

        final_name_list = self.__add_slider_rc.initialize(tmp_ordered_sliders_names, self.handle_combobox_input,
                                                            'Add weight', False, 'Select weight')
        for i, slider_name in enumerate(final_name_list):
            self.__slider_dict[slider_name] = tmp_ordered_sliders_config[i]
        self.addSlider_visibility_test()
        special = 'Vsetky'
        special = self.__add_slider_rc.add_special(special)

    def addSlider_visibility_test(self):
        if len(self.__active_slider_dict) == self.__possible_number_of_sliders:
            self.__add_slider_rc.hide()
        else:
            self.__add_slider_rc.show()

    def create_weight_slider(self, start_neuron: int, end_neuron: int):
        slider_name = 'Vaha {}-{}'.format(start_neuron, end_neuron)
        slider = VariableDisplaySlider()
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider)
        slider.set_variable(self.__weights_reference[start_neuron], end_neuron)
        self.__scroll_area_layout.insertWidget(self.__scroll_area_layout.count()-1, slider)
        self.__active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def create_bias_slider(self, end_neuron: int):
        slider_name = 'Bias {}'.format(end_neuron)
        slider = VariableDisplaySlider()
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider)
        slider.set_variable(self.__bias_reference, end_neuron)
        self.__scroll_area_layout.insertWidget(self.__scroll_area_layout.count() - 1, slider)
        self.__active_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def handle_combobox_input(self, item: tuple):
        self.__disable_update = True
        if item[0] >= 0:
            self.add_slider(item[1])
        else:
            list_of_remaining = self.__add_slider_rc.get_list_of_visible()
            # prve dva su default a vsetky, to treba preskocit
            list_of_remaining = list_of_remaining[1:].copy()
            for name in list_of_remaining:
                self.add_slider(name)
        self.__disable_update = False

    def add_slider(self, slider_name: str):
        if slider_name not in self.__active_slider_dict.keys():
            slider_config = self.__slider_dict[slider_name]
            if slider_config[0]:
                self.create_weight_slider(slider_config[1], slider_config[2])
            else:
                self.create_bias_slider(slider_config[1])
            self.__add_slider_rc.hide_item(slider_name)

    def remove_slider(self, slider_id: str):
        slider = self.__active_slider_dict.pop(slider_id)
        slider.clear()
        self.__add_slider_rc.show_item(slider_id)
        self.addSlider_visibility_test()

    def on_slider_change(self, value):
        self.__controller.wight_bias_change_signal()

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook
app = QApplication(sys.argv)
window = VisualizationApp()
window.showMaximized()

app.exec_()
