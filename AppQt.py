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

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib import backend_bases
from matplotlib.figure import Figure
import matplotlib.animation as animation


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
        # self.__logic_layer = GraphLogicLayer(self)
        self.__keras_model = None
        self.__file_path = '.'
        self.__file_name = ''
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
        self.label = QLabel('Labelik')

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
        #self.__logic_layer.initialize(self.__keras_model)
        self.__open_load_info_label.hide()

    def try_load_points(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Získanie adresy k súboru s hodnotami bodov. Ak nie je načítaný model, zobrazí sa informácia o chybe.
        """
        if self.__keras_model is not None:
            file_path = QFileDialog.getOpenFileName(self, 'Load model input', self.__file_path, 'Text files (*.txt);;CSV files (*.csv)')[0]
            print(file_path)
            return
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
    def __init__(self, parent):
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
        self.__graph_page = parent
        self.__main_graph_frame = MainGraphFrame(parent, self, border=1)
        self.__main_graph_frame.pack(side='bottom', fill='both', expand=True)

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
        self.__number_of_layers = 0

        self.__neural_layers = None

        self.__name_to_order_dict = {}

        self.__order_to_name_dict = {}

        self.__active_layers = []

        # Grafické komponenty. Rozťahovateľné okno.

        horizontal_layout = QHBoxLayout()
        horizontal_layout.setContentsMargins(0, 0, 5, 0)
        horizontal_layout.setSpacing(0)

        options_frame = OptionsFrame()
        options_frame.setMaximumWidth(300)

        self.scrollable_frame = QScrollArea()

        horizontal_layout.addWidget(options_frame)
        horizontal_layout.addWidget(self.scrollable_frame)
        self.__graph_area_layout = QHBoxLayout()
        self.__graph_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__graph_area_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.scrollable_frame.setLayout(self.__graph_area_layout)

        self.__add_remove_layer_rc = RemovingCombobox()
        self.__add_remove_layer_rc.setMaximumWidth(412)
        self.__add_remove_layer_rc.initialize(['layer1', 'layer1'], None, 'Add layer', True, 'Select layer')

        self.__graph_area_layout.insertWidget(0 , self.__add_remove_layer_rc)

        layer_graph = PlotingFrame()
        layer_graph.setMaximumWidth(412)

        self.__graph_area_layout.insertWidget(0, layer_graph)
        # layer_graph.clear()
        #layer_graph.hide()
        #self.__graph_area_layout.removeWidget(layer_graph)


        self.setLayout(horizontal_layout)


    def initialize(self, neural_layers: list, active_layer: list):
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
        self.__options_frame.initialize()

        # Vyčistenie slovníkov.
        self.__order_to_name_dict = {}
        self.__name_to_order_dict = {}

        self.__active_layers = active_layer

        self.__neural_layers = neural_layers
        self.__number_of_layers = len(neural_layers)

        # Vytvorenie predbežného zoznamu názvov vrstiev.
        layer_name_list = []
        for i in range(self.__number_of_layers):
            layer_name_list.append(self.__neural_layers[i].layer_name)

        # Inicializácia AddRemoveComboboxFrame. Funkcia navracia list unikátnych názvov vrstiev.
        # Unikátne názvy su použité ako kľúče v slovníku.
        unique_name_list = self.__add_graph_frame.initialize(layer_name_list, self.show_layer, 'Add layer', True,
                                                                  'Select layer')
        for i, layerName in enumerate(unique_name_list):
            self.__neural_layers[i].layer_name = layerName
            self.__order_to_name_dict[i] = layerName
            self.__name_to_order_dict[layerName] = i

        if len(self.__active_layers) < self.__number_of_layers:
            self.__add_graph_frame.pack(side='right', fill='y', expand=True)

        first_layer_tuple = (self.__neural_layers[0].layer_number, self.__neural_layers[0].layer_name)
        self.show_layer(first_layer_tuple)

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
                layer_to_show.show(self.__scroll_frame.Frame, self.hide_layer, self.show_layer_options_frame)

            layer_to_show.pack(side='left', fill=tk.BOTH, expand=True)
            # Poradové číslo vrstvy je vložené do listu aktívnych vrstiev, ktorý sa využíva pri efektívnejšom updatovaní
            # vykresľovaných grafov.
            self.__active_layers.append(layer_number)
            self.__add_graph_frame.hide_item(layer_name)

            self.__logic_layer.set_points_for_layer(layer_number)
            layer_to_show.apply_changes()
            # Ak je počet aktívnych vrstiev rovný celkovému počtu vrstiev je skrytý panel pre pridávanie nových vrstiev.
            if len(self.__active_layers) == self.__number_of_layers:
                self.__add_graph_frame.pack_forget()

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
            layer.clear()

            self.__active_layers.remove(layer_number)
            self.__add_graph_frame.show_item(layer_name)
            if len(self.__active_layers) < self.__number_of_layers:
                self.__add_graph_frame.pack(side='right', fill='y', expand=True)
            if self.__options_frame.active_layer == layer:
                self.__options_frame.hide_all()

    def show_layer_options_frame(self, layer_number):
        if 0 <= layer_number < self.__number_of_layers:
            layer = self.__neural_layers[layer_number]
            self.__options_frame.initialize_with_layer_config(layer, layer.config)

    def apply_changes_on_options_frame(self):
        self.__options_frame.update_selected_config()

    def update_active_options_layer(self, start_layer):
        self.__options_frame.update_active_options_layer(start_layer)

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


class GraphFrame(QWidget):
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
        #self.__neural_layer = neural_layer
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.__graph = PlotingFrame()
        layout.addWidget(self.__graph)
        #self.__graph = PlotingFrame(self, self, height=420)
        #self.__weight_controller = LayerWeightControllerFrame(self, self)

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
        #self.__parent_controller = graph_frame

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
        #self.__canvas.mpl_connect('button_press_event', self.on_mouse_double_click)
        self.__axis = self.__figure.add_subplot(111, projection='3d')
        self.__draw_3D = False

        backend_bases.NavigationToolbar2.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to  previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        )

        i = 0
        self.__toolbar = NavigationToolbar(self.__canvas, self)
        self.__toolbar.setMinimumHeight(20)
        for child in self.__toolbar.children():
            if not isinstance(child, QLayout):
                if not isinstance(child, QLabel) and not isinstance(child, QWidgetAction):
                    print('schovam', child)
                    child.setVisible(False)
                else:
                    print('ukazuje', child)
                    #child.setVisible(True)



        layout.addWidget(self.__toolbar)

        self.__changed = False
        self.__change_in_progress = False
        self.__locked_view = True
        #self.__ani = animation.FuncAnimation(self.__figure, self.update_changed, interval=105)

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

    def __del__(self):
        print('mazanie graf')

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook
app = QApplication(sys.argv)
window = VisualizationApp()
window.showMaximized()

app.exec_()
