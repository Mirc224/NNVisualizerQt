import cv2
import ntpath
import pandas as pd
import pandas.errors
import matplotlib.colors as mcolors
from os import listdir
from tensorflow import keras
from sklearn.manifold import TSNE
from sklearn import preprocessing
from AdditionalComponents import *
from sklearn.decomposition import PCA
from PlotAndControlComponents import *
from os.path import isfile, join, isdir

BASIC_POINT_COLOR = '#04B2D9'


class GraphLogicLayer:
    def __init__(self, main_graph):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda, starajúca sa o logiku aplikácie. V rámci tejto triedy sa vykonávajú výpočty výstupov na vrstvách po
        zmene váh. Vrstva taktiež updatuje vrstvy, na ktorých došlo k nejakým zmenám. V rámci vrstvy sa taktiež
        načítavajú vstupy do siete a triedy pre jednotlivé body.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__mg_frame:          referencia na triedu MainGraph frame.
        :var self.__input_data:        premenná obashujúca načítané vstupné dáta. Tieto sú zadavané na vstup neurónovej
                                       siete.
        :var self.__points_config:     premenná obsahujúca konfiguračné údaje o vstupných bodoch.
        :var self.__polygon_cords:     obsahuje dovjice bodov predstavujúce začiatočné a koncové body hrán v prípade ak
                                       je v neurónovej sieti možné túto mriežku zobraziť.
        :var self.__number_of_layers:  počet vrstiev neurónovej siete
        :var self.__keras_model:       jedná sa o referenciu na model neurónovej siete načítaný zo súboru, ktorý sa
                                       stará o výpočet výstupov na jednotlivých vrstvách. V rámci používania
                                       aplikácie sa v ňom menia váhy. Pozmenený model je možné aj uložiť.
        :var self.__active_layers:     list obsahujúci poradové čísla jednotlivých vrstiev, neurónovej siete, ktoré sú
                                       aktuálne zobrazované. Referencia na tento list sa nachádza aj v triede
                                       MainGraphFrame.
        :var self.__neural_layers:     list obsahujúci odkazy na inštancie triedy NeuralLayer. Tieto sú v liste zoradené
                                       podľa poradia ako sa vyskytujú v rámci Keras modelu neurónovej siete. Referencia
                                       na tento list sa taktiež nachádza v triede MainGraphFrame.
        :var self.__keras_layers:      list, ktorý odkazuje na vrstvy v Keras modelu. Obsahuje vrstvy na ktorých sa budú
                                       uskutočňovať výpočty. V súčasnosti, keď sú v aplikácií zobrazované všetky vrstvy
                                       neuronónovej siete z Keras modelu, táto premenná nemá veľký význam, no ak by
                                       sme v budúcnosti chceli napríklad zobrazovať len vrstvy neurónovej siete na,
                                       ktorých môže dôjsť k zmene váh, je to možné dosiahnúť pomocou malých úprav
                                       kódu.
        :var self.__changed_layer_q:   zásobnik s unikátnymi id zmenených vrstiev. Sú doň vkladané poradové čísla
                                       vrstiev, na ktorých došlo k zmene váh.
        :var self.__condition_var:     podmienková premenná signalizujúca, že došlo k zmene na niektorej z vrstiev
                                       a je potrebné vykonať aktualizáciu.
        :var self.__monitoring_thread: vlákno sledijúce zmeny váh. Ak dôjde k zmene na niektorej vrstve vlákno vykoná
                                       potrebné výpočty a aktualizuje zobraované body na vrstvách, ktoré boli zmenou
                                       ovplyvnené. Toto vlákno beží počas celého behu aplikácie. Vlákno je potrebné
                                       aby počas výpočtu nedošlo k zaseknutiu apikácie. V hlavnom vlákne teda beží
                                       grafika aplikácie a väčšina výpočtov je ponechaná na toto vlákno.
        :var self.__is_running:        premmenná pre monitorovacie vlákno, ktorá značí, či ešte program beží.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param main_graph: odkaz na inštanciu triedy MainGraphFrame
        """
        # Sú definované základné atribúty. Do premennej self.__mg_frame je priradená hodnota parametra main_graph obsa-
        # hujúca referecniu na inštanciu tejto triedy.
        self.__mg_frame = main_graph
        self.__input_data = None
        self.__points_config = None
        self.__polygon_cords = None

        # Definícia štruktúry siete.
        self.__number_of_layers = 0
        self.__keras_model = None
        self.__active_layers = None
        self.__neural_layers = list()
        self.__keras_layers = list()

        # Vytvorenie, inicializácia a spustenie monitorovacieho vlákna, ktoré sa bude starať vykonávanie výpočtov.
        # Vlákno je nastavené ako deamon vlákno, aby bolo možné program bez problémov vypnúť.
        self.__changed_layer_q = QueueSet()
        self.__condition_var = threading.Condition()
        self.__monitoring_thread = threading.Thread(target=self.monitor_change)
        self.__is_running = True
        self.__monitoring_thread.setDaemon(True)
        self.__monitoring_thread.start()

    def initialize(self, model: keras.Model):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializuje hodnoty pre triedu GraphLogicLayer. Sú vytvorené triedy NeuralLayer na základe poskytnutého Keras
        modelu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param model: načítaný keras model
        """
        # Ak ide o otvorenie nového modelu, sú prípadne staré vrstvy vyčistené, tak aby sa o ne mohol postarať garbage
        # collector.
        for layer in self.__neural_layers:
            layer.clear()

        # Atribútu __keras_model je priradený odkaz na načítaný Keras model, ktorý bol do metódy zaslaný ako argument.
        # Je definovaný počet vrstiev neurónovej siete na základe atribútu načítaného keras modelu. Atribúty
        # neural_layers, keras_layers a active_layers sú inicializované prázdnymi listami. Tieto listy búdú
        # vzápäti napĺňané.
        self.__keras_model = model
        self.__number_of_layers = len(self.__keras_model.layers)
        self.__neural_layers = list()
        self.__keras_layers = list()
        self.__active_layers = list()

        # Keďže po načítaní nového modelu nie sú dostupné žiadne vsutpné dáta, je atribútom polygon_cords a input_data
        # priradená hodnota None.
        self.__polygon_cords = None
        self.__input_data = None

        # Do atribútu je priradená referencia na prázdny dict. Následne sú nastavené hodnoty konfiguračného slovníka
        # pomocou metódy reset_points_config.
        self.__points_config = dict()
        self.reset_points_config()

        # V rámci tohto cyklu sú vytvorené a inicializované inštancie tried zodpovedajúce jednotlivým vrstvám v rámci
        # Keras modelu. Odkazy na vrstvy Keras modelu sú vkladané do listu atribútu keras_layers.
        i = 0
        for layer in self.__keras_model.layers:
            self.__keras_layers.append(layer)
            neural_layer = NeuralLayer(self, layer, i)
            neural_layer.initialize(self.__points_config)
            self.__neural_layers.append(neural_layer)
            i += 1

        # Na záver je inicializovaná aj inštancia triedy MainGraphFrame.
        self.__mg_frame.initialize(self)

    def recalculate(self, starting_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V rámci tejto metódy dochádza k výpočtom výstupov na vrstvách. Aktívnym vrstvám spĺňajúcich podmienku sú
        priradené novo vypočítane hodnoty výstupov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param starting_layer: predstavuje poradové číslo vrstvy, na ktorej došlo k zmene. Keďže ide o klasické a CNN
                               nie o rekurentné neurónove siete, môže zmena váh na vstupe určitej vrstvy ovplyvniť
                               len konkrétnu vrstvu a vrstvy nasledujúce za ňou.
        """
        # Na základe poradového čísla je zo zoznamu aktívnych vrstiev vytvorený zoznam poradových čísel (súčasne aj
        # indexov) vrstiev, ktoré majú byť prepočítané a prekreslené. Prepočítané a prekreslené majú byť vrstvy,
        # ktoré sa nachádzajú v poradi za vrstvou, na ktorej došlo k zmene, teda ich poradové číslo je rovné
        # alebo väčšie ako je hodnota parametra starting_layer. Po vytvorení zoznamu je tento zoznam zasla-
        # ný do metódy, ktorá sa stará o výpočet a priradenie nových hodnôt.
        layers_for_update = [layer_number for layer_number in self.__active_layers if layer_number >= starting_layer]
        self.recalculate_cords(layers_for_update)

        # Zo zoznamu vrstiev, ktoré sa nachádzajú za vrstvou, na ktorej došlo k najskoršej zmene sú ďalej zvolené také
        # vrstvy, na ktorých je aktívna požiadavka na vykreslenie mriežky priestoru. Tento zoznam je následne zaslaný
        # do metódy, ktorá sa o výpočet a priradenie súradníc pre začiatočné a koncové body priestorovej mriežky
        # postará.
        layers_for_update = [layer_number for layer_number in layers_for_update
                             if layer_number >= starting_layer and self.__neural_layers[layer_number].calculate_polygon]
        self.recalculate_grid(layers_for_update)

    def recalculate_cords(self, layers_for_update):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa stará o výpočet výstupov z načítaných vstupov na vrstvách, ktoré boli zaslané ako parameter
        metódy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layers_for_update: list obsahujúci zoznam vrstiev, pre ktoré je potrebné vypočítať a priradiť výstup.
        """

        # Najskôr je potrebné zistiť, či bli nejaké vstupné dáta načítané.
        if self.__input_data is not None:

            # Ak boli vstupné dáta načítané, sú tieto vstupné dáta spolu s listom vrstiev, na ktorých má dôjsť k zmene
            # zaslané ako argumenty do metódy, ktorej návratovou hodnotou sú aktivácie na vrstvách, ktoré do nej boli
            # zaslané. Aktivácie sú v poradí zodpovedajúcom poradiu vrstiev v rámci listu vrstiev, na ktorých má
            # dôjsť k aktualizácií.
            activations = self.get_activation_for_layer(self.__input_data, layers_for_update)

            # Podľa toho, či list obsahoval viac ako jednu vrstvu je navrátená hodnota. Typ návratovej hodnoty môže byť
            # priamo numpy array, ak bol výpočet uskutočnený len pre jednu vrstvu alebo list, ktorý obsahuje výstupy
            # pre jednotlivé vrstvy.
            if not isinstance(activations, list):

                # Ak bola návratová hodnota typu numpy array je táto hodnota obalená do listu, aby ďalšie výpočty boli
                # konzistentné.
                activations = [activations]

            # Hodnota výstupu na jednotlivých vrstvách je postupné priraďovaná príslušným atribútom na príslušných vrst-
            # vách.
            for i, layer_number in enumerate(layers_for_update):
                self.__neural_layers[layer_number].point_cords = activations[i].transpose()

    def recalculate_grid(self, layers_for_update):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa používa na výpočet výstupných súradníc pre začiatočne a koncové body hrán priestorovej mriežky.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layers_for_update: ide o list, ktorý obsahuje zoznam vrstiev, ktoré sú aktívne a je na nich zvolená
                                  možnosť pre zobrazenie mriežky.
        """
        # Skontroluje sa, či existujú nejaké súradnice hrán priestorovej mirežky.
        if self.__polygon_cords is not None:

            # Súradnice hrán priestorovej mriežky sú rozdelené na začiatočné a koncové body a sú poslané ako argument
            # metódy pre výpočet aktivácií na zadaných vrstvách.
            activations_start = self.get_activation_for_layer(self.__polygon_cords[0], layers_for_update)
            activations_end = self.get_activation_for_layer(self.__polygon_cords[1], layers_for_update)

            # Podľa počtu vrstiev v zozname je návratová hodnoa buď list alebo numpy array. Toto je potrebné overiť.
            if not isinstance(activations_start, list):
                # V prípade, že bola zadaná len jedna vrstva, boal návratová hdonota typu numpy array. Preto začiatočné
                # aj koncové výstupné súradnice obalíme do listu, kvôli konzistencií ďalších úkonov v rámci metódy.
                activations_start = [activations_start]
                activations_end = [activations_end]

            # Výpočítané výstupné súradnice hrán sú v cykle priradené atribútom v príslušných vrstiev.
            for layer_number in layers_for_update:

                # Začiatočne a koncové body hrán pre konkrétnu vrstvu musia byť transponované. Z transponovaných bodov
                # je vytvorená dvojica ktorá je priradená atribútu príslušnej neurónovej vrstvy.
                start_points = activations_start[layer_number].transpose()
                end_points = activations_end[layer_number].transpose()
                self.__neural_layers[layer_number].polygon_cords_tuples = [start_points, end_points]

    def monitor_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda je spustená v monitorovaciom vlákne. Vlákno beží počas celého behu programu a ak došlo k zmene váh na
        niektorej z vrstiev, vykoná potrebné výpočty a prekreslí grafy.
        """
        # Nekonečná slučka, v ktorej vlákno beží.
        while True:

            # Zamkne sa sa synchronizčná premenná a testuje sa, či program ešte stále beží.
            self.__condition_var.acquire()
            if not self.__is_running:
                # Ak program už nebeží, je premenná sychnronizačná premenná odomknutá a pomocou return sa ukončí metóda.
                self.__condition_var.release()
                return

            # Následne sa skontroluje, či je zásobník s indexami zmenených vrstiev prázdny alebo nie.
            while self.__changed_layer_q.is_empty():

                # Ak je zásobník prázdny, je zámok odomknutý a vlákno čaka, kým dostane signál o zmene.
                self.__condition_var.wait()

                # Po signalizácií, je zámok opäť zamnkutý a je skontrolované, či je program ešte stále beží, pretože
                # signál môže značiť aj ukončenie programu.
                if not self.__is_running:

                    # Ak je program ukončený je zámok odomknutý a metóda končí.
                    self.__condition_var.release()
                    return

            # Ak sa v zásobníku nachádzajú nejaké vrstvy, na ktorých došlo k zmenám, je obsah zásobníka s týmito
            # vrstvami do pomocnej premennej. Následne je zásobník vyčistený a synchronizačná premenná
            # odomknutá.
            actual_changed = self.__changed_layer_q.copy()
            self.__changed_layer_q.clear()
            self.__condition_var.release()

            # Kvôli efektivite je zo všetkých vrstiev na ktorých došlo k zmenám zvolená najnižšie poradové číslo vrstvy.
            # V cykle sú ešte vrstvám v Keras modele nastavené nové hodnoty váh a biasov.
            starting_layer_number = self.__number_of_layers
            for layer_number in actual_changed:
                if layer_number < starting_layer_number:
                    starting_layer_number = layer_number
                layer = self.__neural_layers[layer_number]
                self.set_layer_weights_and_biases(layer_number, layer.layer_weights, layer.layer_biases)

            # Po nastavení nových váh sú pre vrstvy, ktoré nasledujú v poradí po najnižšej vrstve, na ktorej došlo k
            # zmene váh. Následne sú zmený vysielané jednotlivým vrstávm pomocou metódy broad cast.
            self.recalculate(starting_layer_number)
            self.broadcast_changes(starting_layer_number)

    def get_activation_for_layer(self, input_points, updated_layers_list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vráti aktiváciu pre vrstvy v liste vrstiev.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param input_points:        vstupné body, ktorých aktiváciu chceme získať
        :param updated_layers_list: zoznam vrstiev, pre ktoré majú byť vypočítané hodnty výstupu.
        """
        # Je vytvorený zoznam odkazov na výstupy vrstiev na základe parametra updated_layers_list. Tento zoznam sa nás-
        # ledne použije ako parameter metódy keras modelu.
        calculated_layers = [self.__keras_model.layers[layer_number].output for layer_number in
                             updated_layers_list]

        # Je vytvorený pomocný model, ktorý slúži na výpočet výstupov medziľahlých vrstiev. Pri vytváraní je ako
        # argument použitý vstupn neurónvej siete, ktorá je načítaná a ako výstupy sú použité odkazy na výstupy
        # jednotlivých vrstiev, pre ktoré chceme tieto výstupy vypočítať.
        intermediate_layer_mode = keras.Model(inputs=self.__keras_model.input,
                                              outputs=calculated_layers)

        # Na pomocnom modeli je spusená metóda na predikciu výstupu, ktorej návratovou hodnotou je list výstupných
        # hodnôt bodov po prechode sieťou na určitej vrstve, ktorej výstup sme požadovali.
        return intermediate_layer_mode.predict(input_points)

    def set_points_for_layer(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na výpočet súradníc pre zadanú vrstvu. V rámci tejto metódy sú vypočítané a priradené hodnoty pre výstup
        na základe načítaných vstupov ako aj prípadné výstupné hodnoty pre začiatočne a koncové body hrán v prípade
        ak je na zadanej vrstve zvolená možnosť pre vykresľovanie priestorovej mriežky.

        Parametre:
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, pre ktorú sa má vypočítať aktivácia
        """
        # Ak sú načítané nejaké vsutpné body vykoná sa vypočet a nastavenie výstupov pre zadanú vrstvu.
        if self.__input_data is not None:
            self.__neural_layers[layer_number].point_cords = self.get_activation_for_layer(self.__input_data,
                                                                                           [layer_number]).transpose()

        # Ak je na vrstve zvolená možnosť vykresľovania mriežky, sú vypočítané a nastavnené súradnice začiatočných a
        # koncových bodov.
        if self.__neural_layers[layer_number].calculate_polygon:
            self.set_polygon_cords(layer_number)

    def set_polygon_cords(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa stará o konkrétny výpočet výstupných hodnôt začiatočných a koncových bodov hrán mriežky.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: predstavuje číslo vrstvy, pre ktorú majú byť výstupné hodnoty vypočítané a následne aj
                             priradené.
        """
        # Výpočet aktivácie pre jednotlivé body hrán polygonu.
        if self.__polygon_cords is not None:
            start_points = self.get_activation_for_layer(self.__polygon_cords[0], [layer_number]).transpose()
            end_points = self.get_activation_for_layer(self.__polygon_cords[1], [layer_number]).transpose()
            self.__neural_layers[layer_number].polygon_cords_tuples = [start_points, end_points]

    def broadcast_changes(self, start_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vysiela pre aktívné vrstvy, počnúc vrstvou zadanou ako argument, výzvu na prekreslenie grafu.

        Paramatre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo vrstvy. Vrstvy s poradovým číslom väčším ako je toto, budú prekreslené.
        """

        # Je vytvorený nový zoznam, ktorý bude obsahovať vlákna podieľajúce sa na vypočte zvolených metód na vrstvách
        # neurónovej siete.
        threads = []
        for layer_number in self.__active_layers:
            # Pre každu vrstvu medzi aktívnymi vrstvami, sa otestuje, či je jej poradové číslo väčsie, alebo rovne ako
            # číslo vrstvy, od ktorej sa majú vykonávať zmeny.
            if layer_number >= start_layer:

                # Ak je poradové číslo väčšie, je vytvorené vlákno, ktorému je priradená metóda vrstvy, vykonávajúca
                # prípadný výpočet metódy na redukciu priestoru. Vlákno je pridané do zoznamu pracujúcich vlakien
                # a je spustené.
                thread = threading.Thread(target=self.__neural_layers[layer_number].apply_displayed_data_changes)
                threads.append(thread)
                thread.start()

        # Následne sa prechádza zoznam pracujúcich vlákien a čaká sa, kým každé z nich vykoná zadanú činnosť.
        for thread in threads:
            thread.join()

        # Po tom, čo každé vlákno dokončí svoju úlohu, sú ovplyvnené aktívne vrstvy prekreslené.
        for layer_number in self.__active_layers:
            if layer_number >= start_layer:
                self.__neural_layers[layer_number].redraw_graph_if_active()

        # Pri výpočte mohlo dôjsť k zmenám informácií, napríklad k zmene vyjadrenej variability prislúchajúcej k jedno-
        # tlivým komponentom. V takom prípade je potrebné aktualizovať aj panel možností ak ho prípadné zmeny
        # ovplyvnili.
        self.__mg_frame.update_active_options_layer(start_layer)

    def redraw_active_graphs(self, start_layer=0):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Táto metóda zodpoveda za prekreslenie aktívnych vrstiev, ktoré spĺňaju podmienku, že sa nachádzajú v poradí
        neskôr ako je vrstva, ktorá bola poskytnutá ako parameter.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo vrstvy. Vrstvy, ktoré majú poradove číslo väčšie alebo rovné ako je hodnota
                            tohto parametra, budú prekreslené.
        """
        for layer_number in self.__active_layers:
            if layer_number >= start_layer:
                self.__neural_layers[layer_number].redraw_graph_if_active()

    def set_layer_weights_and_biases(self, layer_number, layer_weights, layer_biases):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá na základe poskytnutého čísla vrstvy a hodnôt váh a biasov, nastaví príslušnej vrstve v keras
        modeli nové hodnoty váh a biasov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number:  poradové číslo vrstvy, ktorej majú byť nastavené nové hodnoty váh a biasov.
        :param layer_weights: obsahuje pole váh určených pre zmenu váh na vrstve modelu.
        :param layer_biases:  obsahuje pole bias hodnôt určených pre zmenu bias hodnôt na vrstve modelu.
        """
        self.__keras_layers[layer_number].set_weights([np.array(layer_weights), np.array(layer_biases)])

    def signal_change_on_layer(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda pridá do zásobníka zmenených vrstiev, číslo vrstvy na ktorej došlo k zmene.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: poradové číslo vrstvy, na ktorej došlo k zmene váh alebo biasov
        """
        self.__condition_var.acquire()
        self.__changed_layer_q.add(layer_number)
        self.__condition_var.notify()
        self.__condition_var.release()

    def handle_images_input(self, input_path):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda načíta obrázkové vstupy na zákalde poskytnutej cesty k obrázku alebo priečinku s obrázkami.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param input_path: cestka k súborom alebo priečinku, na základe ktorje majú byť načítané vstupy
        """

        # Najskôr je potrebné otestovať, či je načítaný nejaký keras model.
        if self.__keras_model is not None:

            # Pre správne načítanie dát je potrebné či sa jedná o list adries k súborom alebo o adresu k priečinku.
            if isinstance(input_path, list):

                # Ak ide o list ciest k súborom, ktoré majú byť načítané, je tento poslaný ako argument do metódy, ktorá
                # podľa zadanej cesty načíta obrázky a zmení im veľkosť podľa vstupných požiadaviek. Načítané a takto
                # upravené obrázky sú vložené do premmennej images_arr_list.
                images_arr_list = self.load_and_reshape_images(input_path, self.__keras_model.input_shape)

                # Podľa toho či je veľkosť listu images_arr_list rozidelna od nuly, je možné určiť, či načítanie
                # obrázkov prebehlo úspešne.
                if len(images_arr_list) != 0:

                    # Ak boli načítané nejaké obrázky, je ich tvar zmenený na tvar, ktorý je požadovaný pre vstup do
                    # neurónovej siete a sú priradené do premennej, ktorá v sebe drží vstupné dáta, ktoré sú
                    # posielané do modelu. Po úspešnom načítaní je potrebné resetovať kofniguračnú premennú
                    # bodov. Následne je v konfiguračnom súbore nastavená pre všetky body základná farba.

                    try:
                        new_value = np.array(images_arr_list).reshape(-1, *self.__keras_model.input_shape[-3:])
                    except Exception as e:
                        return e
                    self.__input_data = new_value
                    self.reset_points_config()

                    basic_color_list = self.__points_config['default_color']

                    # Do premennej, základnú farbu bodov je podľa počtu načítaných vstupov pridaná farba pre každý bod.
                    for _ in range(len(self.__input_data)):
                        basic_color_list.append(BASIC_POINT_COLOR)
                else:

                    # Ak je list prázdny je vrátená chybová hláška, ktorá bude vypísaná na informačnom elemente.
                    return 'No images loaded!'
            else:

                # V tom prípade ak hodnota parametra predstavuje cestu k priečinku obsahujúcemu priečnky s obrázkami
                # podľa tried, je vytvorený nový zoznam, ktorý obsahuje názvy priečinkov. Taktiež sú definované
                # pomocné premenné, slúžiace k správnemu načítaniu a priradeniu tried.
                only_dirs = [file_name for file_name in listdir(input_path) if isdir(join(input_path, file_name))]
                input_data_tmp = []
                labels_tmp = []
                for dir_name in only_dirs:

                    # Z názvov priečinkov a poskytnutej cesty do nadradeného priečinku je vytvorená cesta do priečinku.
                    # Priečinok je prehľadaný a sú vybrané názvy všetkych obrázkových súborov typu .png, .jpg, .jpeg.
                    # V závislosti od názvu obrázkového súboru je zostrojený zoznam absolútnych ciest k týmto
                    # súborom.
                    dir_path = join(input_path, dir_name)
                    image_path_list = [join(dir_path, file_name) for file_name in listdir(dir_path)
                                       if isfile(join(dir_path, file_name)) and
                                       join(dir_path, file_name).lower().endswith(('.png', '.jpg', '.jpeg'))]

                    # Zoznam vytvorených absolútnych ciest je zaslaný do metódy, ktorá sa postará o načítanie vstupov
                    # a zmenu ich rozmeru podľa požadovaného rozmeru pre vstup neuronovej siete.
                    try:
                        images_arr_list = self.load_and_reshape_images(image_path_list, self.__keras_model.input_shape)
                    except Exception as e:
                        return e

                    # Do listu, ktorý drží triedy doposiaľ načítaných bodov, je pridaný názov triedy v závislosti od
                    # počtu načítaných obrázkov. Ku listu všetkých dát je pripočítaný list načítaných dát z
                    # aktuálneho priečinku.
                    labels_tmp += [dir_name for _ in range(len(images_arr_list))]
                    input_data_tmp += images_arr_list

                # Po prejdení všetkých vnorených priečinkov, je potrebné zistiť, či boli načítané nejaké obrázky.
                if len(input_data_tmp) != 0:

                    # Ak boli nejaké vstupy načítané je ich tvar prevedený na tvar, ktorý je požadvoaný pre vstup do
                    # neurónovej siete. Takto zmenené body sú priradené do premennej držiacej načítané vstupné
                    # údaje. Následne je resetovaná konfiguračná premenná pre body a do pomocnej premennej je
                    # priradený odkaz na list, ktorý bude obsahovať základnú farbu načítaných vstupov.
                    self.__input_data = np.array(input_data_tmp).reshape(-1, *self.__keras_model.input_shape[-3:])
                    self.reset_points_config()
                    basic_color_list = self.__points_config['default_color']

                    # Do konfiguračného súboru je každému načítanému bodu priradená základná farba.
                    for _ in range(len(self.__input_data)):
                        basic_color_list.append(BASIC_POINT_COLOR)

                    # Do konfiguračnej premennej je priradený odkaz zoznamu tried zodpovedajúcich načítaným bodom.
                    # Ďalej je do konfiguračnej premennej priradená aj farba bodov priradená v závislosti od
                    # prislušnosti bodu k určitej triede.
                    self.__points_config['label'] = labels_tmp
                    self.__points_config['label_color'] = self.get_label_color_list(labels_tmp)
                else:

                    # Ak neboli načítané žiadne obrázky, je vrátená chybová hláška. Táto hláška sa vypíše na informačnom
                    # prvku v hornej časti grafického rozhrania aplikácie.
                    return 'No images loaded!'

            # Následne sú načítané hodnoty prepočítané a priradené zobrazeným vrstvám. Sú nastavené aj zákaldné zobra-
            # zené súradnice, pretože pri načítaní by mohlo dôjsť k chybe, kvôli požiadavke na zobrazenie súrandice
            # ktorá pri nových dátach nemusí existovať a to najmä pri metóde PCA, pri ktorej počet zobrazených
            # komponentov záleží aj od počtu načítaných bodov.
            self.recalculate()
            self.set_default_cords_for_each_layer()
            self.broadcast_changes()

            # Následne je na panel možností aplikovaná aktualizácia, pretože v závisloti od načítaných bodov sa mohli
            # sprístupniť nové možnosti pre vzhľad grafu.
            self.__mg_frame.apply_changes_on_options_frame()
            return None
        else:

            # Ak ešte žiaden model nebol načítaný, je vrátená chybová hláška, ktorá bude zobrazená pomocou informačného
            # prvku vo vrchnej časti grafického rozhrania aplikácie.
            return 'No model loaded!'

    def load_points(self, filepath):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pomocou tejto metódy je možné načítať súbory obsahujúce číselné premenné. Na vstup je možné použiť súbory typu
        .txt alebo .csv.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: cesta k súboru obashujúcemu vstupy.
        """
        # Najskôr sa overí, či je načítaný keras model.
        if self.__keras_model is not None:

            # Ak je model načítaný, na zákalde poskytnutej cesty sa získa koncovka súboru, aby bolo možné načítať súbory
            # v závislosti od typu. V súboroch .txt musia byť vstupné hodnoty odelené medzerou. Pri súboroch typu .csv,
            # je požadované, aby hodnoty boli oddelné čiarkou. Súbory nesmú obsahovať hlavičku. Následne sú pomocou
            # funkcie z knižnice pandas súbory načítané. Ak pri načitávaní došlo k chybe, je znenie tejto hodnoty
            # vrátené a zobrazené pomocou informačného prvku.
            file_ext = ntpath.splitext(filepath)[1]
            if file_ext == '.txt':
                try:
                    data = pd.read_csv(filepath, sep=' ', header=None)
                except pd.errors.ParserError as e:
                    return e
            else:
                try:
                    data = pd.read_csv(filepath, header=None)
                except pd.errors.ParserError as e:
                    return e

            # Z načítaného keras modelu je zistený požadovaný vstupný rozmer dát.
            shape_of_input = self.__keras_model.layers[0].input_shape[1]

            # Rozmer načítaných vstupných dát je porovnaný s požadovaným rozmerom.
            if len(data.columns) == shape_of_input:

                # Ak vstupné dáta spĺňajú požadovaný rozmer, sú testované jednotlivé stĺpce načítaných dát. Pomocou fun-
                # kcie je kontorolované, či sú všetky hodnotych v stĺpcoch číselne.
                is_column_numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).to_list()
                if False in is_column_numeric:

                    # V prípade, že nie sú všetky hodnoty v stĺpcoch číselné, je vrátená chybová hláška, ktorou je pou-
                    # žívateľ oboznámeny s týmto faktom.
                    return 'Data columns contains non numeric values!'

                # Po uistení sa, že sú všetky načítané hodnoty číselné, sú dáta prevedené na numpy array a priradneé do
                # premennej držiacej odkaz na vstupné dáta. Taktiež je resetovaná konfiguračná premenná pre body.
                self.__input_data = data.to_numpy()
                self.reset_points_config()

                # Všetkým novo načítaným bodom je v konfiguračnom súbore priradená základná farba.
                points_color = self.__points_config['default_color']
                points_color.clear()
                for _ in range(len(self.__input_data)):
                    points_color.append(BASIC_POINT_COLOR)

                # Ak je počet stĺpcov načítaných dát v rozmedzí 1 až 4, sú definované hrany priestorovej mriežky.
                if 1 < shape_of_input < 4:

                    # Ak sa počet stĺpcov nachádza v prípustnom intervale, je medzi stĺpcami nájdená maximálna a
                    # minimálna hodnota vstupu, na základe ktorých bude definovaná priestorová mriežka. Každá
                    # hrana bude rozdelená na päť častí.
                    minimal_cord = np.min(self.__input_data[:, :shape_of_input], axis=0).tolist()
                    maximal_cord = np.max(self.__input_data[:, :shape_of_input], axis=0).tolist()
                    polygon = Polygon(minimal_cord, maximal_cord, [5, 5, 5])

                    # Vrcholy a pdobne aj hrany mriežky sú prevedené na numpy array a sú priradené do premennej.
                    polygon_peak_cords = np.array(polygon.Peaks)
                    edges_tuples = np.array(polygon.Edges)

                    # Začiatočné a koncové vrcholy sú získane na základe definovaných hrán a priradené do premennej
                    # držiacej odkaz na začiatočne a koncové súradnice vrcholov.
                    self.__polygon_cords = []
                    self.__polygon_cords.append(polygon_peak_cords[:, edges_tuples[:, 0]].transpose())
                    self.__polygon_cords.append(polygon_peak_cords[:, edges_tuples[:, 1]].transpose())

                    # Pre každú vrstvu je nastavené, že je možné priestorovú mriežku zobraziť.
                    for layer in self.__neural_layers:
                        layer.possible_polygon = True
                else:

                    # V prípade ak vstup nespĺňa podmienky, je každej vrstve atribút signalizujúci možnosť vykreslenia
                    # priestorovej mriežky nastavený na False.
                    for layer in self.__neural_layers:
                        layer.possible_polygon = False

                # Ak prebehlo načítvanaie bez chyby, sú aplikované zmeny a grafy všetky grafy sú prekreslené. Taktiež
                # je aktualizovaný aj panel možností, pretože načítaním dát sa môžu sprístupniť určité nové možnosti.
                self.recalculate()
                self.set_default_cords_for_each_layer()
                self.broadcast_changes()
                self.__mg_frame.apply_changes_on_options_frame()
                # V prípade, že načítavanie prebehlo úspešne, je navrátená hodnota None.
                return None
            else:

                # Ak rozmer načítaných vstupných dát nie je rovnaký ako sieťou požadovaný, je vrátená chybová správa
                # pomocou ktorej je používateľ o tejto skutočnsti informovaný.
                return 'Different input point dimension!'
        else:

            # V prípade ak nie je načítaný žiadny model, je vrátená chybová hláška, ktorá používateľa informuje.
            return 'No Keras model loaded!'

    def load_labels(self, filepath):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda uskutoční načítanie tried zo súboru a ich priradenie už načítaným bodom. Triedy budú priraďované v poradí
        v akom boli vstupy načítavané.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param filepath: parameter predsavuje cestu k súboru obshaujúceho triedy pre každý z načítaných bodov.
        """

        # Najskôr sa skontroluje, či je načíaný keras model.
        if self.__keras_model is not None:

            # Nasleduje overenie, boli načítané body, ktorým majú byť triedy priradené.
            if self.__input_data is None or len(self.__input_data) == 0:

                # Ak nie sú načítané žiadne body je vrátená chybová hláška, ktorá je následne zobrazeá používateľovi.
                return 'No points loaded!'

            # Následne sú dáta načítané pomocou funkcie z knižnice pandas.
            try:
                data = pd.read_csv(filepath, header=None)
            except Exception as e:
                return e

            # Súbor musí obsahovať iba hodnoty tried prislúchajúce jednotlivým bodom, preto je potrebné skontorlovať
            # počet načítnaých stĺpcov.
            if len(data.columns) > 1:

                # Ak je počet stĺpcov väčší ako je 1, je vstup chybný a je informácia o chybe je vrátená, aby mohla byť
                # zobrazená používateľovi.
                return 'Invalid number of columns!'

            # Následne je skontrolovaný počet načítaných tried.
            if len(data[0].values) != len(self.__input_data):

                # Ak je počet načítaných tried odlišný od počtu načítaných vstupov, nastala chyba a chybová hláška je
                # vrátená, aby bol používateľ o vyskytnutej chybe informovaný.
                return 'Invalid number of labels!'

            # Po kontrolách sú načítané hodnoty tried priadené do konfiguračnej premennej bodov. Hodnoty konfiguračnej
            # premennej súvisiace s farbou triedy a aktívnych ofarbených bodov sú vyčistené.
            self.__points_config['label'] = data[0].values.tolist()
            self.__points_config['different_points_color'] = list()
            self.__points_config['active_labels'] = list()

            # Do konfiguračnej premennej je do hodnoty pre farbu bodu podľa príslušnosti k triede priradený výstup z
            # metódy, ktorý sa o vytvorenie unkátneho ofarbenia postará.
            self.__points_config['label_color'] = self.get_label_color_list(data[0].tolist())

            # Na záver sú prekreslené grafy a aktualizovaný je aj panel možností, na ktorom sa mohli sprístupniť nové
            # možnosti.
            self.broadcast_changes()
            self.__mg_frame.apply_changes_on_options_frame()

    def get_label_color_list(self, label_list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vytvorí list obsahujúci farby pre každý bod v závislosti od príslušnosti k určitej triede.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param label_list: zoznam tried.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: vracia list farieb, pre jednotlivé body v závislosti od ich triedy
        """
        # Z určitých farieb podporovnaných knižnicou matplotlib je vytvorený zoznam unikátnych farieb. Na začiatok a
        # teda medzi prvé priradené farby sú vložené názvy farieb zo základnej palety. Je však odstránená biela
        # farba, pretože by body neboli vidieť. Následne sú prejdené aj ďalšie zo zoznamu farieb podporovnaých
        # matplotlibom.
        possible_colors = list(mcolors.BASE_COLORS.keys())
        possible_colors.remove('w')
        for name, value in mcolors.CSS4_COLORS.items():

            # Postupne sú prechádzané všetky farby a hexadecimálna hodnota je prevedená na desiatkové číslo, ktoré sa
            # následne porovná s hodnotou, ktorá určuje úroveň svetlosti. Ak odtieň farby spĺňa kritérium, je meno
            # farby vložené do zoznamu možných farieb.
            if int(value[1:], 16) < 15204888:
                possible_colors.append(name)

        # Z poskytnutého zoznamu tried získame unikátne triedy. Do premennej je uložená informácia o počte unikátnych
        # farieb, ktoré je možne použiť. Je definovaná premenná typu dict, ktorá bude obsahovať ako kľúč príslušnu
        # triedu a bude sa odkazovať na príslušnú farbu.
        unique_labels = set(label_list)
        label_color_dict = {}
        number_of_unique_colors = len(possible_colors)
        for i, label in enumerate(unique_labels):

            # Na základe jedntolivých tried je dictionary naplnený hodnotami. Unikátnych farieb nie je neobmedzený
            # počet, preto v prípade veľkého počtu tried je pridané ošetrenie, ktoré zabráni chybe no začne
            # priraďovať už použité farby.
            label_color_dict[label] = possible_colors[i % number_of_unique_colors]

        # Je vytvorená pomocná premenná, ktorá bude niesť farbu pre každý bod.
        color_label = []
        for label in label_list:

            # V rámci cyklu je podľa príslušnosti k triede bodu priradená farba.
            color_label.append(label_color_dict[label])

        return color_label

    def reset_points_config(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda restuje hodnoty konfiguračnej premennej.
        """
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
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Na základe poskytnutého zoznamu aboslutnych ciest k obrázkom tieto obrázky načíta a zmení ich veľkosť v závislo-
        sti od vstupného rozmeru.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param img_path_list:
        :param input_shape:

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: vracia pole naplnené obrázkami prevdenými na polia čísel
        """

        # Je definovaná premenná, do ktorej budú vkladané načítané a upravené obrázky. Na základe vstupného rozmeru sú
        # priradené hodnoty o požadovanej výške a šírke obrázku, ktoré budú pouźité pri zmene rozmeru. Taktiež je
        # preddefinovaná hodnota príznakov.
        list_of_images_arrays = []
        img_height = input_shape[-3]
        img_width = input_shape[-2]
        flags = cv2.IMREAD_COLOR
        # Preddefinovaná odnota príznaku hovorí, že ide o farebný obrázok, no na základe vstupného rozmeru je potrebné
        # to zistiť. Zistíme to na základe posledného čísla, ktoré pojednáva o počte vstupných kanálov.
        if input_shape[-1] == 1:
            # Ak je posledné číslo 1, znamená to, že sa jedná o čiernobiely vstup, preto je nastavený tento príznak.
            flags = cv2.IMREAD_GRAYSCALE
        for img_path in img_path_list:

            # Postupne sú zo zoznamu na základe cesty načítané jednotlivé obrázky a ich rozmer je zmenený podľa
            # požiadavky. Takto transformované obrázky sú pridané do premennej, ktorá drží všetky doposiaľ
            # načítané obrázky.
            img_array = cv2.imread(img_path, flags=flags)
            new_array = cv2.resize(img_array, (img_height, img_width))
            list_of_images_arrays.append(new_array)

        # Kompletný zoznam načítaných obrázkov je použitý ako návratová hodnota metódy.
        return list_of_images_arrays

    def set_default_cords_for_each_layer(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Táto metóda nastaví každej vrstve siete základné súradnice zobrazované pre jednotlivé metódy. 
        """
        for layer in self.__neural_layers:
            layer.set_default_cords()

    def require_options_bar_update(self, neural_layer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vykoná aktializáciu panelu možností na zákalde poskytnutého čísla vrstvy.
        
        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer: odkaz na vrstu na ktorej došlo k zmene.
        """
        self.__mg_frame.update_layer_if_active(neural_layer)

    def __del__(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Deštruktor triedy, pri končení nastaví hodnotu premennej definujúcu, či program stále beží a notifikuje bežiace
        vlákno.
        """
        self.__condition_var.acquire()
        self.__is_running = False
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
    def __init__(self, logic_layer: GraphLogicLayer, keras_layer: keras.layers.Layer,
                 layer_number: int):
        """        
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda predstavuje vrstvu neurónovej siete. Je zodpovedná za výpočty metód na redukciu priestoru, na nastavova-
        nie zobrazovaných súradníc a taktiež slúži ako spojovací článok medzi triedou GraphPage obsahujúcou ovládač
        a vykresľovaný graf a logikou aplikácie.

        Aribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__layer_name:     názov vrstvy získaný z Keras modelu. Tento názov môže byť v priebehu práce s modelom
                                    zmenený za účelom dosiahnutia unikátneho mena, ktoré sa stane identifikátorom
                                    vrstvy.
        :var self.__has_f_maps:     poskytuje informáciu o tom, či sa na danej vrstve vyskytujú feature mapy.
        :var self.__selected_fm:    číslo zvolenej feature mapy, pre ktorú sa má zobrazovať výstup.
        :var self.__output_shape:   výstupný tvar vrstvy.
        :var self.__output_dim:     tento atribút nesie informáciu o výstupnej dimenzií vrstvy.
        :var self.__num_of_out:     predstavuje počet výstupov z vrstvy. Môže zodpovedať počtu výstupných feature map
                                    ale aj počtu neurónov na ďalšej vrstve ak ide o huste prepojenú neurónovú sieť.
        :var self.__layer_weights:  drží hodnotý váh na danej vrstve pre výstuné prvky.
        :var self.__layer_biases:   obsahuje hodnoty bias pre výstupné elementy.
        :var self.__has_points:     nesie informáciu o tom, či sú v rámci vrstvy priradené body.
        :var self.__calc_polygon:   informuje o tom, či majú byť na vrstve počítané hodnoty pre zobrazenie priestorovej
                                    mriežky.
        :var self.__poly_cords_t:   atribút môže obsahovať usporiadané dvojice pre začiatočne a koncové body hrán pries-
                                    torovej mriežky, ktoré sú transformované pre zobrazenie na tejto vrstve.
        :var self.__layer_config:   konfiguračná premenná, ktorá obsahuje informácie o nastaveniach vrstvy.
        :var self.__points_config:  referencia na konfiguračnú premennú s nastaveniami načítaných bodov.
        :var self.__graph_frame:    môže obsahovať odkaz na triedu GraphFrame zodpovednú za grafické zobrazenie výstupu
                                    vrstvy.
        :var self.__logic_layer:    odkaz na triedu typu GraphLogicLayer.
        :var self.__layer_number:   poradové číslo vrstvy.
        :var self.__point_cords:    obsahuje výstupné hodnoty bodov na tejto sieti bez aplikácie niektorej z metód na
                                    redukciu priestoru.
        :var self.__used_cords:     list použitých súradnic, ktorého obsah závisí od použitej metódy redukcie priestoru.
        :var self.__activ_nodes_l:  názvy výstupov, používaných pri výpise informácií ohľadom použitej PCA, konkrétne
                                    informácie o vplyve výstupu na jednotlivé hlavné komponenty.
        :var self.__pc_labels:      obsahuje preddefinované nazvy pre hlavné komponenty.
        :var self.__method_cords:   obsahuje súradnice bodov po použití metódy na redukciu priestoru. Pri zmene zobraze-
                                    ných súradníc vďaka tomu nie je potrebné prepočítať na výstupy pre metódu a dané
                                    súradnice ale z tejto premennej sú vytiahnuté len požadované súradnice.
        :var self.__visible:        nesie informáciu o tom, či je pre danú vrstvu vykresľovaný zobrazený rámec s grafom
                                    a ovládačom váh.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param logic_layer:  odkaz na logicku vrstvu GraphLogicLayer.
        :param keras_layer   odkaz na prislúchajúcu vrstvu z keras modelu.
        :param layer_number: poradové číslo vrstvy.
        """
        # Na základe prijatého parametra obsahujúceho odkaz na vrstvu modelu keras je nastavné príslušné meno vrstvy. 
        # Taktiež sú predefinované niektoré hodnoty atribútov.
        self.__layer_name = keras_layer.get_config()['name']
        self.__logic_layer = logic_layer
        self.__layer_number = layer_number
        self.__has_f_maps = False
        self.__selected_fm = 0
        self.__output_shape = keras_layer.output_shape
        self.__output_dim = None
        self.__layer_config = {}
        self.__has_points = False
        self.__calc_polygon = False
        self.__poly_cords_t = None
        self.__points_config = None
        self.__graph_frame = None
        self.__point_cords = np.array([])
        self.__used_cords = []
        self.__activ_nodes_l = []
        self.__pc_labels = []
        self.__method_cords = []
        self.__visible = False

        # Podľa toho, či výstupný tvar obsahuje viac ako dva rozmery, či su na výstupe vrstvy prítomné feature mapy.
        # Výstupný formát má tvar (veľkosť vstupnej dávky (väčšinou hodnota None), (tu sa možu nachádzať dva rozme-
        # ry v prípade ak ide o vstup viacrozmerných dát, napríklad obrázkov, ak ide o obrázky je na prvom mieste
        # výška obrazku a na druhom mieste šírka obrázku), posledné číslo udáva počet výstupov prípadne počet
        # feature máp)
        if len(self.__output_shape) > 2:

            # Ak  výstupný rozmer obsahuje viac ako 2 položky (batch size a počet výstupov), je zrejmé, že sa jedná
            # o vrstvu, ktorá má na svojom výstupe feature mapy. Preto je nastavená hodnota atribútu has_f_maps na
            # hodnotu True. Taktiež je vypočítaná výstupná dimenzia, ktorú získame vynásobením výšky a šírky.
            self.__has_f_maps = True
            self.__output_dim = self.__output_shape[-3] * self.__output_shape[-2]
        else:

            # Ak je počet prvkov vo výstupnom rozmere je menej ako dva ide o vrstvu, na výstupe ktorej sa nenachádzajú
            # feature mapy, preto sa ponecháva prednastavená hodnota atribútu has_f_maps na hodnote False. Podľa
            # počtu výstupov je nastavená výstupná dimenzia.
            self.__output_dim = self.__output_shape[-1]

        # Počet výstupov (čiže neurónov na ďalšej vrstve) alebo počet vytvorených feature máp na výstupe je získany
        # na základe rozmeru výstupu vrstvy.
        self.__num_of_out = self.__output_shape[-1]

        # Následne sa zisťuje, či sú na vrstve prítomné nastaviteľné vrstvy a to pomocou zistenia dĺžky listu, vráteného
        # z metódy keras vrstvy. Tento list bude väčší ako 0 ak sú na vrstve meniteľné váhy.
        if len(keras_layer.get_weights()) != 0:

            # Do atribútov sú priradené kópie váh a biasov príslušnej vrstvy z keras modelu.
            self.__layer_weights = keras_layer.get_weights()[0].copy()
            self.__layer_biases = keras_layer.get_weights()[1].copy()
        else:

            # Ak vrstva neobsahuje žiadne meniteľné váhy alebo biasy, je atribúttom priradená hodnota None.
            self.__layer_weights = None
            self.__layer_biases = None

    def initialize(self, points_config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda inicializuje atribúty triedy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param points_config: odkaz na konfiguračnú premennú obsahujúcu informácie a nastavenia ohľadom zobrazovaných
                              bodov
        """
        # Je vyčistený konfiguračný atribút vrstvy obsahujúci nastavenia. Ak boli v minulosti načítané nejaké body, je
        # im atribútu, ktorý na ne ukazuje priradená hodnota None. Na zákalde poskytnutého parametra, je priradený
        # do atribútu points_config odkaz na konfiguračnú premennú zobrazovaných bodov.
        self.__layer_config = {}
        self.__point_cords = None
        self.__points_config = points_config

        # Sú definované základné označenia osí. Do atribútu pc_labels, ktorý bude obsahovať mená možných hlavných
        # komponentov je priradený prázdny zoznam. Podobne aj do atribútu method_cords, ktorý by bude obsahovať
        # výstupné hodnoty bodov na základe použitej metódy je priradený čistý zoznam. Je definovaná pomocná
        # premenná axis_labels, ktorá bude obsahovať označenia osí podľa v závislosti od počtu zobraziteľ-
        # ných výstupov.
        axis_default_names = ['Label X', 'Label Y', 'Label Z']
        axis_labels = []
        self.__pc_labels = []
        self.__method_cords = []

        # Na základe toho, či vrstva na svojom výstupe obsahuje feture mapy sú zoznamy pre označenie hlaných komponentov
        # a prvkov vplývajúcich na jednotlivé hlavné komponenty naplnené názvami.
        if self.__has_f_maps:

            # Ak vrstva obsahuje na svojom výstupe feature mapy je na základe výstupného tvaru do pomocných premenných
            # priradený počet riadkov a počet stĺpcov jednotlivých feature map.
            number_of_rows = self.__output_shape[1]
            number_of_cols = self.__output_shape[2]

            # Podľa počtu riadkov a stĺpcov, je vytvorený cyklus priraďujúci označenia.
            for height_i in range(number_of_rows):
                for width_i in range(number_of_cols):
                    # Body feature map vplývajúce na hlavné komponenty sú nazývane tak, že v ich názve figuruje číslo
                    # riadku a číslo stĺpca, v ktorom sa hodnota nachádza. Označenia hlavných komponentov sú
                    # mapované na zákalde riadku a stĺpca. Maximálny možný počet komponentov je rovanký ako
                    # je počet hodnôt v jednej feature mape.
                    self.__activ_nodes_l.append(f'Node {height_i}-{width_i}')
                    self.__pc_labels.append(f'PC{height_i * number_of_cols + width_i + 1}')
        else:

            # Ak vrtva neobsahuje feature mapy, tak sú v rámci jedného cyklu priradené názvy pre hlavné komponenty a
            # jednotky vplývajúce na výstup.
            for i in range(self.__num_of_out):
                # Jednotky sú označované po poradí.
                self.__activ_nodes_l.append(f'Node {i}')
                self.__pc_labels.append(f'PC{i + 1}')

        # Na základe výstupnej dimenzie je potrebné zistiť, koľko súradníc je možné vykresliť. To získame pomocou
        # funkcie, ktorá zistí minimum z čísla 3 (maximálne môžeme zobraziť 3D graf) a počtu výstupov.
        number_of_cords = min(3, self.__output_dim)
        for i in range(number_of_cords):
            # Podľa počtu zobraziteľných súradníc sú do listu pre označenie osí pridané základné názvy z pomocného listu
            # podľa poradového čísla.
            axis_labels.append(axis_default_names[i])

        # Sú definované nastavenia a informačné hodnoty do konfiguračnej premennej. Tieto hodnoty sú používané pri voľbe
        # zobrazovaných súradnic, voľbe metódy na redukciu priestoru a na základe týchto hodnôt sú zobrazené možnosti
        # na panale možností, keď je zvolená možnosť nastavení pre príslušnú vrstvu.

        # Hodnota pod kľúčom has_feature_maps informuje o tom, či výstup z vrstvy pozostáva z feature máp. Pod kľúčom
        # output shape je uložený výstupný tvar vrstvy. Na základe hodnoty apply_changes sa vyhodnocuje, či je
        # potrebné aplikovať zmeny. Pomocou cords_changed je prenášaná informácia o tom, či boli zmenené sú-
        # radnice a preto je potrebné prenastaviť body zobrazované v grafe. Pod kľúčom output_dimension je
        # možné nájsť výstupnú dimenziu vrstvy. Hodnota premennej layer_name, ako už názov napovedá obsa-
        # huje názov vrstvy. Na základe max_visible dim je možné zistiť, akú dimenziu je možné maximál-
        # ne zobraziť. Axis labels obsahuje odkaz na zoznam označení osí grafu. Number_of_samples po-
        # skytuje informáciu o počte načítaných bodov.
        self.__layer_config['has_feature_maps'] = self.__has_f_maps
        self.__layer_config['output_shape'] = self.__output_shape
        self.__layer_config['apply_changes'] = False
        self.__layer_config['cords_changed'] = False
        self.__layer_config['output_dimension'] = self.__output_dim
        self.__layer_config['layer_name'] = self.__layer_name
        self.__layer_config['max_visible_dim'] = number_of_cords
        self.__layer_config['axis_labels'] = axis_labels
        self.__layer_config['number_of_samples'] = 0

        # Nasledujúce hodnoty v rámci konfiguračnej premennej obsahujú informácie o tom, čo a ako má byť v grafe zobra-
        # zené. Pod kľúčom possible_polygon sa nachádza hodnota pojednávajúca o tom, či môže byť na vrstve vykreslená
        # priestrová mriežka. Color_labels informuje o tom, či majú byť body grafu ofarbené v závislosti od prísluš-
        # nosti k triede. Show_polygon pojednáva o tom či v prípade, ak je možné vykresliť na danej vrstve priesto-
        # rovú mriežku má byť táto mriežka zobrazená.
        self.__layer_config['possible_polygon'] = False
        self.__layer_config['color_labels'] = False
        self.__layer_config['show_polygon'] = False
        self.__layer_config['locked_view'] = False

        # Podľa toho, či je počet súradníc, ktoré možno zobraziť rovný trom, je do kofnigruačnej premennej priradená
        # ifnormácia o tom, či je možné prepínať medzi 2D a 3D zobrazením
        if number_of_cords >= 3:

            # Ak počet prípustných zobrazených súradníc je rovný 3, je do konfiguračného súboru zapísané, že je povolené
            # prepínať medzi 2D a 3D zorbazením.
            self.__layer_config['draw_3d'] = True
        else:

            # Ak počet prírpustných zobrazených súradníc nie je 3, používateľ nemá možnosť prepínať medzi 2D a 3D zobra-
            # zením.
            self.__layer_config['draw_3d'] = False

        # Je definováná aktuálne použitá metóda na redukciu priestoru, ktorá je uložená pod kľúčom used_method. Pod kľú-
        # čom config_selected_method je informácia pre panel možností, ktorý nesie informáciu o tom, na ktorú metódu
        # používateľ naposledy klikol.
        self.__layer_config['used_method'] = 'No method'
        self.__layer_config['config_selected_method'] = 'No method'

        # V nasledujúcej časti sú definované konfigurácie, ktoré v sebe nesú nasavenia pre výpočet a zobrazenie metód
        # na redukciu priestoru. Najskôr sú definované hodnoty, ktoré bude táto konfiguračná peremenná obsahovať.
        # Konfigruacia pre metódu na redukciu No method obsahuje zoznam súradníc, ktoré majú byť po výpočte zo-
        # brazené. Konfigurácia pre metódu PCA obsahuje zoznam súradníc, ktoré majú byť zobrazené, počet prí-
        # pustných hlavných komponentov a odkazy na pandas dataframe a series inštanice, ktoré obsahujú in-
        # formácie o výstupe metódy PCA.
        no_method_config = {'displayed_cords': None}
        pca_config = {'displayed_cords': None,
                      'n_possible_pc': 0,
                      'percentage_variance': None,
                      'largest_influence': None,
                      }

        # Metóda t-SNE je určená pre vizualizáciu viac dimenzionálnych dát. Dáta dokáže transformovať maximálne do troch
        # dimenzií. Toto číslo získame na základe výsledku funkcie na získanie minima z počtu výstupných dimenzií a
        # čísla 3.
        number_t_sne_components = min(self.__output_dim, 3)

        # Konfigurácia pre metódu t-SNE pozostáva z viacerých častí. Prvá čast obsahuje parametre pre jej výpočet.
        # Hodnoty týchto parametrov boli nastavené podľa dokumentácie k metóda t-SNE v knižnici sklearn.
        used_config = {'n_components': number_t_sne_components,
                       'perplexity': 30,
                       'early_exaggeration': 12.0,
                       'learning_rate': 200,
                       'n_iter': 1000}

        # Parametre bude používateľ schopný meniť a preto tieto vstupy bude potrebné ošetriť a zistiť či sú valídne,
        # preto je vytvorený slovník obsahujúci pod identifikátorom parametra požadovaný typ a hranice prípustné
        # pre príslušný parameter. Vďaka tomu, že sú kľúče zhodné s názvami parametrov, bude možné načítané
        # vstupy v rámci cyklu jednoducho skontrolovať.
        parameter_borders = {'n_components': (1, int, number_t_sne_components),
                             'perplexity': (0, float, float("inf")),
                             'early_exaggeration': (0, float, 1000),
                             'learning_rate': (float("-inf"), float, float("inf")),
                             'n_iter': (250, int, float("inf"))
                             }

        # Jedodtlivé časti konfigurácie pre metódu t-SNE sú priradené do premennej. Premenná obsahuje aj kľúč options_pa
        # -rameter, pod ktorým sa skrývajú hodnoty, ktoré používateľ navolil v paneli možností no možno ich ešte nepou-
        # žil.
        t_sne_config = {'used_config': used_config,
                        'options_config': used_config.copy(),
                        'parameter_borders': parameter_borders,
                        'displayed_cords': None}

        # Do konfiguračného súboru sú pridelené konfigurácie jednotlivých metód.
        self.__layer_config['no_method_config'] = no_method_config
        self.__layer_config['PCA_config'] = pca_config
        self.__layer_config['t_SNE_config'] = t_sne_config

        # Na záver je pre všetky metódy nastaviť valídne základné súradnice, ktoré majú byť zobrazené.
        self.set_default_cords()

    def apply_displayed_data_changes(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda nastaví dáta, ktoré sa majú zobraziť, ak náhodou došlo k zmenám metód na redukciu priestoru na danej vr-
        stve.
        """
        # Podľa toho, či sú načítané nejaké vstupy sa buď to vykoná požadovaná metóda na redukciu alebo metóda skončí.
        if self.__has_points:

            # Ak boli načítané nejaké vstupy je podľa použitej metódy na redukicu priestoru použitá metóda na jej výpo-
            # čet.
            used_method = self.__layer_config['used_method']
            if used_method == 'No method':
                self.apply_no_method()
            elif used_method == 'PCA':
                self.apply_PCA()
            elif used_method == "t-SNE":
                self.apply_t_SNE()

            # Na zákalde zvolenej metódy na redukciu priestoru sú nastavené súradnice, ktoré sa majú byť zobrazené.
            self.set_used_cords()
            # Na záver sú triede zodpovednej za vykreslenie výstupu vrstvy nastavené požadované súradnice na zobrazenie.
            self.set_points_for_graph()

    def set_points_for_graph(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Podľa použitej metódy nastaví zobrazované údaje. Ak je nie je použitá žiadna metóda na redukciu a na vrstve je
        zvolená možnosť pre vykreslenie priestorovej mriežky, sú nastavené aj hodnoty začiatočných a koncových bodov
        hrán.
        """
        used_method = self.__layer_config['used_method']
        self.set_displayed_cords()
        if used_method == 'No method':
            if self.__poly_cords_t is not None:
                if self.__graph_frame is not None:
                    self.set_displayed_cords_for_polygon()

    def set_used_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Z konfigruačnej premennej na základe použitej metódy na redukciu priestoru získa súradnice, ktoré sa majú na
        výstupe majú zobraziť. Tie sú uložené do atribútu used_cords.
        """
        used_method = self.__layer_config['used_method']
        if used_method == 'No method':
            self.__used_cords = self.__layer_config['no_method_config']['displayed_cords']
        elif used_method == 'PCA':
            self.__used_cords = self.__layer_config['PCA_config']['displayed_cords']
        elif used_method == 't-SNE':
            self.__used_cords = self.__layer_config['t_SNE_config']['displayed_cords']

    def set_displayed_cords_for_polygon(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Podľa požadovaných súradníc na zobrazenie sú vybraté súradnice začiatočných a koncových bodov priestorovej mrie-
        žky. Z týchto bodov je následne vytvorený list súradníc.
        """
        # Na základe zoznamu požadovaných súradníc su pre jednotlivé začiatočne a koncové body zvolené príslušné hodno-
        # ty. Tieto sú transponované
        tmp1 = self.__poly_cords_t[0][self.__used_cords].transpose()
        tmp2 = self.__poly_cords_t[1][self.__used_cords].transpose()
        self.__graph_frame.plotting_frame.line_tuples = list(zip(tmp1, tmp2))

    def set_displayed_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pre vykresľovaný graf, (ak je vytvorená inštancia tejto triedy) sú nastavené hodnoty, ktoré chce používateľ zo-
        braziť. Nastavenie týchto hodnôt závisí od typu výstupy z vrstvy.
        """
        # Skontroluje sa, či existuje inštancia, ktorej majú byť nastavené hodnoty pre zobrazenie.
        if self.__graph_frame is not None:
            # Ak áno testuje sa o aky typ výstupu ide.
            if self.__has_f_maps and self.__layer_config['used_method'] == 'No method':
                # Ak sa na výstupe nachádzajú feature mapy a nie je použitá žiadna z metód na redukciu priestoru, test-
                # uje sa, či sú nastavené súradnice pre zobrazenie.
                if len(self.__method_cords) != 0:
                    # Ak sú súradnice nastavené, na základe  atribútu so zvolenou feature mapou je zvolený výstup
                    # zodpovedajúcí aktivácií načítaných bodov pre danú feature mapu. Z hodnôt výstupov jedno-
                    # tlivých obrázkov pre túto feature mapu sú zvolené zadané súradnice.
                    feature_map_points = self.__method_cords[self.__selected_fm, :, :, :].transpose()
                    self.__graph_frame.plotting_frame.points_cords = feature_map_points[:, self.__used_cords[0],
                                                                     self.__used_cords[1]].transpose()
            else:
                # V prípade že na výstupe vrstvy sa nachádzajú feature mapy, alebo je použitá niektorá z metód na red-
                # ukciu priestoru, skontorluje sa, či sú zadané súradnice na zobrazenie pre metódu.
                if len(self.__method_cords) != 0:
                    # Ak sú súradnice nastavené, sú grafu priradené hodnoty, ktoré majú byť zobrazené. Tieto hodnoty
                    # závisia od metódy a taktiež požadovaných zobrazených hodnôt pre túto metódu.
                    self.__graph_frame.plotting_frame.points_cords = self.__method_cords[self.__used_cords]

    def apply_no_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá ak nie je nastavená žiadna ina metóda na redukciu priestoru. Spočíva v jednoduchom
        priradení hodnôt výstupu vrstvy vypočítaných na zákalde predikcie neurónovej siete.
        """
        self.__method_cords = self.__point_cords.copy()

    def apply_PCA(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vykoná výpočet metódy PCA. Postup pri výpočte sa v niektorých častiach líši s ohľadom na to, či je výstup
        z vrsty vo forme feature máp alebo nie.
        """
        # Pri vykonávaní metódy môže dôjsť k tomu, že nastane delenie nulou. Vtedy by bolo vypísané varovanie. Pomocou
        # tohto príkazu je tento varovný výpis potlačený.
        np.seterr(divide='ignore', invalid='ignore')

        # Spracovanie bodov pred použitím metódy závisí od toho či su výstupom vrstvy feature mapy alebo nie.
        if self.__has_f_maps:

            # Ak sa na výstupe vrstvy nachádzajú feature mapy, sú podľa zvolenej feature mapy vybrané príslušné feature
            # mapy pre každý zo vstupov. Hodnoty sú získane z atribútu obsahujúce výstupy pre danú vrstvu, získané na
            # základe predpovede keras modelu. Tieto feature mapy, väčšinou vo forme polí, sú potom pre jednotlivé
            # vstupy transformované na vektory a vložené do premennej.
            feature_map_points = self.__point_cords[self.__selected_fm, :, :, :].transpose()
            points_cords = np.array([xi.flatten() for xi in feature_map_points])
        else:

            # Ak výstup neobsahuje feature mapy, sú body pre metódu získané jednoducho prekopríovaním hodnôt získaných
            # získaných predikciou výstupu pre danú vrstvu pomocou keras modelu.
            points_cords = self.__point_cords.transpose().copy()

        # Načítané body je potrebné následne normalizovať. Na to je použitá funkcia z knižnice sklearn. Normalizácia je
        # potrebná, pretože metóda by vyšším hodnotám prikladala väčší význam čo sa týka vysvetlenej variability, aj
        # keď by to nemusela byť pravda v prípade keď budú dáta štandardizované.
        scaled_data = preprocessing.StandardScaler().fit_transform(points_cords)
        pca = PCA()

        # Normalizvané dáta sú vložené do funkcie, ktorá zistí hodnoty vlastných vektorov na základe týchto dát.
        # Potom ako sú hodnoty vlastných vektorov nastavené, sú pomocou nich dáta transformované a priradené
        # do pomocnej premennej. Dáta v tejto pomocnej premennej zobrazujú transformované súradnice bodov
        # prislúchajúce jednotlivým hlavným komponentom.
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)

        # Súradnice sú následne transponované a priradené do atribútu obsahujúce výstupné hodnoty po aplikácií niektorej
        # z metód na redukciu priestoru.
        self.__method_cords = pca_data.transpose()

        # Do pomocnej premennej priradíme počet vzniknutých hlavných komponentov.
        number_of_pcs_indexes = pca.explained_variance_ratio_.size
        if number_of_pcs_indexes > 0:
            # Ak boli vypočítané niektoré hlavné komponenty, sú informácie získané na základe výstupu z metódy priradené
            # do konfiguračného súboru PCA metódy. Tieto informácie môžu byť následne zobrazené v panele možností.
            # Najskôr je vytvorená pandas séria obsahujúca informácie o tom, koľko variability vysvetľujú jedno-
            # tlivé hlavné komponenty. Séria má index podľa mena jednotlivých komponentov. Prvý komponent vždy
            # vysvetľuje najviac variability. Ako posledné je vytvorený pandas dataframe, ktorý obsahuje info-
            # rmácie o vplyve prvkov výstupu na jednotlivé hlavné komponenty. Indexy tohoto dataframe sú pri-
            # radené na zákalde atribútu, obsahujúceho názvy pre tieto prvky.
            self.__layer_config['PCA_config']['percentage_variance'] = pd.Series(
                np.round(pca.explained_variance_ratio_ * 100, decimals=1),
                index=self.__pc_labels[:number_of_pcs_indexes])
            self.__layer_config['PCA_config']['largest_influence'] = pd.DataFrame(pca.components_.transpose(),
                                                                                  index=self.__activ_nodes_l)

    def apply_t_SNE(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa použije ak bola na redukciu priestoru použitá metóda t-SNE.
        """
        # Najskôr je do pomocnej premennej priradený dictionary, ktorý obsahuje hodnoty parametrov na výpočet tejto
        # metódy.
        t_sne_config = self.__layer_config['t_SNE_config']
        # Podľa toho, či sa na výstupe nachádzajú feature mapy sú do pomocnej premennej priradené hodnoty predpovedaných
        # výstupov.
        if self.__has_f_maps:

            # Ak sa na výstupe nachádzajú feature mapy, tak na zákalde zvolenej feature mapy sú vybraté aktivácie pre 
            # zodpovedajúce tejto feature mape pre všetky body. Metóda t-SNE pracuje len s vektormi, preto je potre-
            # bné jednotlivé výstupy preransformovať a vložiť do pomocnej premennej.
            feature_map_points = self.__point_cords[self.__selected_fm, :, :, :].transpose()
            points_cords = np.array([xi.flatten() for xi in feature_map_points])
        else:

            # Ak sa na výstupe feature mapy nenáchádzajú sú body transponované aby príznaky prislúchajúce jednotlivým 
            # bodom boli zoradené v stĺpcoch.
            points_cords = self.__point_cords.transpose().copy()
        # Následne je inicializovaná trieda pre výpočet metódy t-SNE na základe použitej konfigurácie. Následne je usku-
        # točnená metóda t-SNE a výstup je transponovaný a vložený do atríbútu držiaceho súradnice, ktoré majú byť
        # zobrazené.
        tsne = TSNE(**t_sne_config['used_config'])
        transformed_cords = tsne.fit_transform(points_cords).transpose()
        self.__method_cords = transformed_cords

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vyčistí grafické zobrazenie neurónovej vrstvy, ak je nejaké zobrazované.
        """
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None

    def signal_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda posúva signal o zmene váhy logickej vrstve.
        """
        self.__logic_layer.signal_change_on_layer(self.__layer_number)

    def set_polygon_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V rámci tejto metódy je vyžiadané nastavenie výstupných súradníc zodpovedajúcich začiatočným a koncovým bodom
        priestrovoej mriežky.
        """
        self.__logic_layer.set_polygon_cords(self.__layer_number)
        self.apply_displayed_data_changes()

    def require_graphs_redraw(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vyžiada prekreslenie všetkých aktívnych grafov.
        """
        self.__logic_layer.redraw_active_graphs()

    def redraw_graph_if_active(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Ak je vytvorená grafická reprezentácia výstupu vrstvy je pomocou inštancie triedy PlottingFrame.
        """
        if self.__graph_frame is not None:
            self.__graph_frame.redraw_graph()

    def use_config(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Ak je vytvorená grafická reprezentácia výstupu vrstvy je pre inštanciu tejto triedy použitá nastavená konfigu-
        rácia.
        """
        if self.__visible:
            # V závislosti od toho, či na vrstve došlo k zmenám, na základe ktorých je potrebne vykonať nové výpočty sú
            # vykonané nasledujúce časti kódu.
            if self.__layer_config['apply_changes']:
                # Ak boli vykonané zmeny, ktoré vyžadujú uskutočnenie výpočtov, pokračcuje s touto vetvou. Po vykonaní
                # výpočtov sú príznakové premenné opäť nastavené na False.
                self.apply_displayed_data_changes()
                self.__layer_config['cords_changed'] = False
                self.__layer_config['apply_changes'] = False
            elif self.__layer_config['cords_changed']:

                # V prípade ak došlo k zmenám len ohľadom zobrazovaných súradníc, nie je potrebné vykonávať nové výpočty
                # pretože výstupne hodnoty všetkých súradníc sa nachádzajú v atribúte method cords. Pomocou tohto atri-
                # bútu sú nové súradnice zobrazené. Na záver je príznaková premenná opäť nastavená na False.
                self.set_used_cords()
                self.set_points_for_graph()
                self.__layer_config['cords_changed'] = False

            # Nová konfigurácia je následne nastavená inštancií na zobrazovanie výstupov.
            self.__graph_frame.apply_config(self.__layer_config)

    def create_graph_frame(self, options_command, hide_command):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V tejto metóde je vytvorená a inicializovaná inštancie triedy GraphFrame, ktorá slúži na vykreslenie výstupu
        vrstvy a ovládanie váh.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param options_command: odkaz na funkciu, ktorá sa má zavolať po kliknutí na tlačidlo options v rámci inštancie
                                triedy GraphFrame.
        :param hide_command:    odkaz na funkciu, ktorá sa má zavolať po kliknutí na tlačidlo hide v rámci inštancie
                                triedy GraphFrame.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return návratová hodnota tejto metódy je odkaz na inštanciu triedy GraphFrame, ktorá obsahuje ovládač váh
                a graf na vykrslovanie
        """
        # Ak by náhodou bola inštancia už vytvorená, je táto vyčistiená referencia na ňu je zrušená.
        if self.__graph_frame is not None:
            self.__graph_frame.clear()
            self.__graph_frame = None
        # Následne je vytvorená nová inštancia triedy GraphFrame a je inicializovaná. Do metódy inicializácie je záslaný
        # odkaz na danú vrstvu neurónovej siete a taktiež odkaz na funkcie, ktoré sa majú zavolať po kliknutí tlačidiel
        # Options a Hide.
        self.__graph_frame = GraphFrame(self.__has_f_maps)
        self.__graph_frame.initialize(self, options_command, hide_command)

        self.__visible = True
        return self.__graph_frame

    def set_default_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V rámci tejto metódy sú nastavené základné súradnice, ktoré majú byť zobrazené pri použití jednej z metód na
        redukciu priestoru.
        """
        # Z konfiguračnej premennej je zistený počet načítaných vstupov a dimenzia výstupu vrstvy. Na základe výstupnej
        # dimenzie vrstvy je zistený maximálny možný počet zobrazených súradníc. Ten je získaný ako minimu z hodnoty 3
        # a počtu výstupných dimenzií.
        n_of_samples = self.__layer_config['number_of_samples']
        o_dimension = self.__layer_config['output_dimension']
        number_of_cords = min(3, o_dimension)
        if self.__has_f_maps:
            # Ak vrstva na výstupe obsahuje feature mapy, je potrebné nastaviť súradnice, ktoré majú byť použite pre
            # zobrazenie bez použitia metódy na redukciu priestoru, odlišne ako v iných prípadoch, pretože pri
            # tomto prípade sú súradnice kvôli konfortu zadávané v inom formáte.
            predefined_cords = [[], []]
            # Podľa možného počtu zobrazených súradníc sú preddefinované hodnoty, ktoré majú byť v rámci feature máp zo-
            # brazené.
            for i in range(number_of_cords):
                x_cord = i
                y_cord = i
                # Je potrebné ošetriť aby neboli feature mapám nastavené nesprávne indexy. To by mohlo nastať napríklad
                # ak by mala feature mapa tvar 2x1. Defaultne sú súradnice priradzované po diagonále. V prípade že by
                # mala byť porušená podmienka je na miesto použitá posledná možná súradnica.
                if x_cord >= self.__output_shape[-3]:
                    x_cord = self.__output_shape[-3] - 1
                if y_cord >= self.__output_shape[-2]:
                    y_cord = self.__output_shape[-2] - 1
                predefined_cords[0].append(x_cord)
                predefined_cords[1].append(y_cord)
        else:

            # V prípade ak sa na výstupe feature mapy nenachádzajú, sú jednoducho pre zobrazenie zvolené prvé 3 výstupné
            # údaje.
            predefined_cords = []
            for i in range(number_of_cords):
                predefined_cords.append(i)

        # Následne sú definované aj základné súradnice pre ostatné metódy.
        self.__layer_config['no_method_config']['displayed_cords'] = predefined_cords
        # Prípustné zobrazené súradnice metódy PCA závisia od toho, aká je výstupná dimenzia, aký je počet načítaných
        # vstupov, no sú limitované aj tým, že je možne zobraziť maximálne 3D graf. Na základe týchto podmienok je
        # preto vytvorený list súradníc na základe týchto obmedzení.
        self.initialize_default_PCA_cords()
        # Pri metóde t-SNE je situácia takmer podobná ako pri metóde PCA, no počet možných súradníc je však priamo nas-
        # tavený na základe parametra na výpočet metódy t-SNE.
        self.initialize_default_tSNE_cords()

    def initialize_default_PCA_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví v rámci konfiguračnej premennej prípustné zobrazované súradnice pre metódu PCA.
        """
        self.__layer_config['PCA_config']['displayed_cords'] = list(range(min(self.__layer_config['number_of_samples'],
                                                                              self.__layer_config['output_dimension'],
                                                                              3)))

    def initialize_default_tSNE_cords(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví v rámci konfiguračnej premennej prípustné zobrazované súradnice pre metódu t-SNE.
        """
        self.__layer_config['t_SNE_config']['displayed_cords'] = list(range(
                                                    self.__layer_config['t_SNE_config']['used_config']['n_components']))

    def signal_feature_map_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá pri zmene feature mapy. Pre túto feature mapu sú nastavené zobrazované súradnice a je
        prekreslený aj graf. Ak je vrstva práve zobrazená na panele možností, je aktualizovaný aj panel možností.
        """
        self.apply_displayed_data_changes()
        self.redraw_graph_if_active()
        self.__logic_layer.require_options_bar_update(self)

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
        return self.__poly_cords_t

    @property
    def possible_polygon(self):
        return self.__layer_config['possible_polygon']

    @possible_polygon.setter
    def possible_polygon(self, value):
        self.__layer_config['possible_polygon'] = value

    @polygon_cords_tuples.setter
    def polygon_cords_tuples(self, new_cords_tuples):
        self.__poly_cords_t = new_cords_tuples
        if self.__poly_cords_t is not None:
            self.__layer_config['possible_polygon'] = True
        else:
            self.__layer_config['possible_polygon'] = False

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
        return self.__calc_polygon

    @calculate_polygon.setter
    def calculate_polygon(self, value):
        self.__calc_polygon = value

    @property
    def graph_frame(self):
        return self.__graph_frame

    @graph_frame.setter
    def graph_frame(self, value):
        if value is not None:
            self.__visible = True
        else:
            self.__visible = False
        self.__graph_frame = value

    @property
    def number_of_outputs(self):
        return self.__num_of_out

    @property
    def output_shape(self):
        return self.__output_shape

    @property
    def output_dimension(self):
        return self.__layer_config['output_dimension']

    @property
    def selected_feature_map(self):
        return self.__selected_fm

    @selected_feature_map.setter
    def selected_feature_map(self, new_value):
        self.__selected_fm = new_value


class GraphFrame(QFrame):
    def __init__(self, has_feature_maps, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Obaľovacia trieda obsahujúca inštanciu triedy zodpovednej za vykresľovanie výstupov na príslušnej vrstve a 
        inštanciu ovládača váh pre danú vrstvu. Výstup tejto vrstvy je zobrazovaný pomocou triedy PlottingFrame a 
        váhy na tejto vrstve je možné meniť pomocou inštancie triedy LayerWeightControllerFrame.
        
        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__neural_layer:   odkaz na inštanciu triedy NeuralLayer. 
        :var self.__has_fmaps:      informácia o tom, či sú na výstupe vrstvy dostupné feature mapy.
        :var self.__options_btn:    odkaz na inštanciu grafického prvku QPushButton, ktorý slúži na zobrazenie možnosti
                                    vrstvy na panele možností.
        :var self.__hide_btn:       odkaz na inštanciu grafického prvku QPushButton, ktorý slúži na skrytie grafického
                                    zobrazenia danej vrstvy.
        :var self.__graph:          inštancia triedy PlottingFrame, ktorá zabezpečuje vykresľovanie poskytnutých hodnôt.
        :var self.__feature_map_cb: podľa toho, či sú na výstupe vrstvy prítomné feature mapy, táto premenná obsahuje
                                    odkaz na inštanciu triedy QComboBox alebo hodnotu None. Tento combobox zobrazuje,
                                    ktorá feature mapa je zobrazovaná na výstupe
        :var self.__w_controller:   odkaz na ovládač váh na prislušnej vrstve. Odkazovať sa na inštanciu triedy v
                                    závislosti od toho, či sú na výstupe vrstvy feature mapy alebo nie.
        
        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param has_feature_maps: bool hodnota, nesúca informáciu o tom, či sú na výstupe vrstvy prítomné feature mapy.
        :param args: argumenty pre konštruktor predka.
        :param kwargs: keyword argumenty pre konštruktor predka.
        """
        # Je zavolaný konštruktor predka.
        super(GraphFrame, self).__init__(*args, **kwargs)
        # Sú definované a incializované atribúty.
        self.__neural_layer = None
        self.__has_fmaps = has_feature_maps
        self.__options_btn = QPushButton()
        self.__hide_btn = QPushButton()
        self.__graph = PlottingFrame()

        # Podľa toho, či sú na výstupe feature mapy alebo nie je do atribútu w_controller priradená inštancia príslušnej
        # triedy a definovaná hodnota atibútu feature_map
        if self.__has_fmaps:

            # Ak sa na výstupe vrstvy feature mapy nachádzajú je atribútu feature_map_cb priradený odkaz na inštanciu
            # triedy QComboBox. Do atribútu w_controller je priradený odkaz na inštanciu triedy
            # FMWeightControllerFrame.
            self.__feature_map_cb = QComboBox()
            self.__w_controller = FMWeightControllerFrame()
        else:

            # Ak sa na výstupe vrstvy feature mapy ne nachádzajú je atribútu feature_map_cb priradená hodnota None.
            # Do atribútu w_controller je priradený odkaz na inštanciu triedy NoFMWeightControllerFrame.
            self.__feature_map_cb = None
            self.__w_controller = NoFMWeightControllerFrame()
        self.init_ui()

    def init_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia grafických prvkov.
        """
        # Oknu je nastavená maximálna šírka 500 pixelov. Predkovy je priradené objektové meno na zákalde ktorého je mu
        # priradené vonkajšie orámovanie.
        self.setMaximumWidth(500)
        self.setObjectName('graphFrame')
        self.setStyleSheet("#graphFrame { border: 1px solid black; } ")

        # Je vytvorené hlavné rozmiestnenie. Je nastavený zobrazovaný text na oboch tlačidlách. Na vrchu hlavného roz-
        # miestnenia je definované ďalšie rozmiestnenie, v ktorom budú ležať tlačidlá pre zobrazenie možností na pa-
        # nele možností a pre skrytie vrstvy. Podľa toho, či sú na výstupe definované feature mapy je do rozmiest-
        # nenia vložený aj ComboBox, držiaci zoznam feature máp, ktoré je možné zobraziť. Na záver sú do hlav-
        # ného rozmiestnenia priadné inštancie triedy PlottingFrame a WeightControllerFrame.
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        buttons_wrapper_layout = QHBoxLayout()
        buttons_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons_wrapper_layout)
        self.__options_btn.setText('Options')
        buttons_wrapper_layout.addWidget(self.__options_btn, alignment=QtCore.Qt.AlignLeft)
        if self.__has_fmaps:
            buttons_wrapper_layout.addWidget(self.__feature_map_cb)
        self.__hide_btn.setText('Hide')
        buttons_wrapper_layout.addWidget(self.__hide_btn, alignment=QtCore.Qt.AlignRight)
        self.setLayout(layout)
        layout.addWidget(self.__graph)
        layout.addWidget(self.__w_controller)

    def initialize(self, neural_layer: NeuralLayer, options_command=None, hide_command=None):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V závislosti od príslušnej neurónovej vrstvy sú nastavené hodnoty jednotlivých atribútov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer:    odkaz na neurónovú vrstvu, ktorej výstup má zobrazovať a ktorej bude možné meniť váhy.
        :param options_command: odkaz na funkciu, ktorá sa má zavolať pri kliknutí na tlačidlo Options.
        :param hide_command:    odkaz na funkciu, ktorá sa má zavolať pri kliknutí na tlačidlo Hide.
        """
        self.__neural_layer = neural_layer
        if options_command is not None:

            # Ak bol poskytnutý odkaz na funkciu option_command je táto funkcia pripojená na signál, ktorý sa vyvolá
            # po kliknutí na toto tlačidlo. Funkcia je obalená do lambda funkcie, aby bolo možné poskytnúť aj
            # argument funkcie, konrétne číslo príslušnej vrstvy, aby ju bolo možné identifikovať.
            self.__options_btn.clicked.connect(lambda: options_command(self.__neural_layer.layer_number))
        if hide_command is not None:

            # Ak bol poskytnutý odkaz na funkciu hide_command je táto funkcia pripojená na signál, ktorý nastane, ak
            # bolo na tlačidlo kliknuté. Funkcia je obalená v lambda funkcií, aby bolo možné, do funkcie zodpovednej
            # za skrytie vrstvy, zaslať identifikátor v podobe čísla príslušnej vrstvy.
            self.__hide_btn.clicked.connect(lambda: hide_command(self.__neural_layer.layer_number))

        # Následne je inicializovaná inštancia PlottingFrame, ktorej parametre získame z poskytnutej inštancie triedy
        # NeuralLayer.
        self.__graph.initialize(self, neural_layer.output_dimension, neural_layer.points_cords,
                                neural_layer.points_config, neural_layer.layer_name)

        # Podľa toho, či sú na výstupe vrtvy prítomné feature mapy, sú incializované kontrolery a prípadne naplnený
        # Combobox.
        if self.__has_fmaps:
            # Ak sú na výstupe vrstvy prítomne feature mapy, znamená to, že bol definovaný ComboBox, ktroý je potrebné
            # naplniť prípustnými hodnotami. Najskôr je ComboBox vyčistený a je získaný výstupný tvar neurónovej
            # vrstvy.
            self.__feature_map_cb.clear()
            output_shape = neural_layer.output_shape

            # Z výstupného tvaru je možné zistiť počet výstupných feature máp. Na základe tohto údaju je v cykle
            # naplnený combobox.
            for feature_map_index in range(output_shape[-1]):
                self.__feature_map_cb.addItem('Feature map {}'.format(feature_map_index))
            # Je nastavená funkcia, ktorá sa zavolá po vybratí nejakej z hodnôt ComboBoxu.
            self.__feature_map_cb.currentIndexChanged.connect(self.initialize_selected_feature_map)
            # Na záver je inicializovaná inštancia triedy FMWeightControllerFrame.
            self.__w_controller.initialize(self, neural_layer.layer_weights, neural_layer.layer_biases,
                                                self.__feature_map_cb.currentIndex())
        else:
            # Je inicializovaná inštancia triedy NoFMWeightControllerFrame.
            self.__w_controller.initialize(self, neural_layer.layer_weights, neural_layer.layer_biases)

    def initialize_selected_feature_map(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá je zavolaná pri požiadavke na zmenu zobrazovanej feature mapy.

        """
        if self.__has_fmaps:
            # Je zistený index prvku, na ktorý bolo v rámci Comboboxu kliknuté a keďže sú v ňom prvky zoradené, index
            # predstavuje feature mapu, ktorá ma byť zobrazená. Tento údaj je zaslaný do kontroléra, ktorý na
            # základe neho zmení zobrazované vahý na váhy prisluchajúce danej feature mape. Údaj o zvolenej
            # feature mape je nastavený aj atribútu príslušnej neurónovej vrstvy a následne je zavolaná
            # funkcia, ktorá vykoná zmeny súvisiace so zmenou zobrazovanej feature mapy.
            selected_fm = self.__feature_map_cb.currentIndex()
            self.__w_controller.initialize_for_fm(selected_fm)
            self.__neural_layer.selected_feature_map = selected_fm
            self.__neural_layer.signal_feature_map_change()

    def weight_bias_change_signal(self):
        """
        Popis
        --------
        Posúva signál o zmene váhy neurónovej vrstve.
        """
        self.__neural_layer.signal_change()

    def redraw_graph(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda ktorá zaistí aby došlo k prekresleniu grafu.
        """
        self.__graph.redraw_graph()

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá slúži na vyčistenie inštancie triedy a taktiež kompozitných tried. Jej úlohou je uvoľniť používané
        prostriedky. Na záver metódy je explicitne zavolaný garbge collector.
        """
        self.__graph.clear()
        self.__graph = None
        self.__w_controller.clear()
        self.__graph = None
        self.deleteLater()
        gc.collect()

    def require_graphs_redraw(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Posúva signal nadradenej vrstve, signalizujúci potrebu prekreslenia všetkých grafov.
        """
        self.__neural_layer.require_graphs_redraw()

    def apply_config(self, config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje poskytnutú konfiguráciu na inštanciu PlottingFrame.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param config: príslušná konfigurácia, ktorá má byť aplikovaná na inštanciu PlottingFrame.
        """
        # Priestorovú mriežku je možné zobraziť, iba ak nie je použitá žiadna metóda redukcie priestoru. Ak je táto
        # možnosť zvolená, je vykonaná len v závislosti od toho, aká metóda je použitá. Pri metódach PCA a t-SNE
        # je vykresľovanie mriežky zakázané.
        if config['used_method'] == 'No method':
            self.__graph.draw_polygon = config['show_polygon']
        else:
            self.__graph.draw_polygon = False
        self.__graph.locked_view = config['locked_view']
        self.__graph.graph_labels = config['axis_labels']
        self.__graph.is_3d_graph = config['draw_3d']
        self.__graph.set_color_label(config['color_labels'])

    @property
    def plotting_frame(self):
        return self.__graph


class MainGraphFrame(QWidget):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Hlavné okno podstránky s grafmi. Okno vytvorí scrollovacie okno, do ktorého budú následne vkladané jednotlivé vrstvy
    s grafmi a ich ovládačmi. Vytvorí taktiež options okno, ktoré slúži na zobrazovanie informácii o zvolenem bode a
    možnosti nastavenia zobrazenia grafov v jednotlivých vrstvách.

    Atribúty
    --------------------------------------------------------------------------------------------------------------------
    :var self.__logic_layer:        obsahuje odkaz na vrstvu, ktorá sa zaoberá výpočtami a logickou reprezentáciou
                                    jednotlivých vrstiev a váh.
    :var self.__number_of_layers:   počet vrstiev načítanej neurónovej siete
    :var self.__neural_layers:      obsahuje referenciu na vrstvy neurónovej siete.
    :var self.__name_to_order_dict: každej vrstve je postupne prideľované poradové číslo vrstvy, tak ako ide od
                                    začiatku neurónovej siete. Ako kľúč je použitý názov vrstvy.
    :var self.__order_to_name_dict: podľa poradia je každej vrstve pridelené poradové číslo. Poradové číslo je kľúčom
                                    do dict. Jeho hodnotami sú názvy jedntlivých vrstiev.
                                    Ide o spätný dict k __name_to_order_dict.
    :var self.__active_layers:      referencia na list, ktorý obsahuje poradové čísla aktívnych vrstiev. Je zdieľaný
                                    s logickou vrstvou aplikácie.
    :var self.__scrollable_frame:   Qt skrolovacie okno, obalom pre QtWidget scrollbar content.
    :var self.__scrollbar_content:  udržuje v sebe jednotlivé GraphFrame, ktoré obsahujú grafyy a ovládače váh.
    :var self.__graph_area_layout:  rozloženie komponentov v scrollbar_content widgete.
    :var self.__options_frame:      odkaz na okno s možnosťami zobrazenia vrstvy.
    :var self.__add_rm_layer_rc:    combobox na výber vrstvy. Po vybratí sa vrstva zobrazí a odstráni z comboboxu.
    """
    def __init__(self, *args, **kwargs):
        super(MainGraphFrame, self).__init__(*args, **kwargs)

        # Definovanie základných atribútov triedy.
        self.__logic_layer = None
        self.__number_of_layers = 0

        self.__neural_layers = None

        self.__name_to_order_dict = {}

        self.__order_to_name_dict = {}

        self.__active_layers = []

        # Definovanie grafických komponentov triedy.
        self.scrollable_frame = QScrollArea(self)

        self.__scrollbar_content = QWidget(self)

        self.__graph_area_layout = QHBoxLayout(self.__scrollbar_content)

        self.__options_frame = OptionsFrame(self)

        self.__add_rm_layer_rc = RemovingCombobox(self.__scrollbar_content)

        # Inicializácia grafických komponentov
        self.initialize_ui()

        # Skrytie panelu s možnosťami.
        self.__options_frame.hide_option_bar()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia, rozloženie a zobrazenie grafických komponentov.
        """
        # Základné rozmiestnenie v rámci wrappera hlavného okna.
        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.setContentsMargins(0, 0, 5, 0)
        horizontal_layout.setSpacing(0)

        # Nastavenie fixnej šírky pre panel s možnosťami. K fixnej šírke sme pristúpili z dizajnérskeho hľadiska. Pri
        # inej politike nastavovania šírky sa panel zväčšuje a zmenšuje podľa rozkliknutej karty, čo zanecháva
        # zlý grafický dojem. Daná šírka je príjemná na pohľad a väčšina grafických komponentov sa do nej
        # bez problémov zmestí.
        self.__options_frame.setFixedWidth(320)

        # Do hlavného rozmiestnenia sú pridané hlavné položky a to okno s možnosťami a hlavné okno pre zobrazovanie
        # grafov. Panel s možnosťami sa bude nachádzať na ľavej strane a scrollovacie okno bude zobrazené napravo.
        horizontal_layout.addWidget(self.__options_frame, alignment=QtCore.Qt.AlignLeft)
        horizontal_layout.addWidget(self.scrollable_frame)
        self.__options_frame.adjustSize()

        # Nastavenie parametrov pre skcrollovacie okno a nastavenie rozmiestnenie v rámci skrolovacieho okna.
        # Nové prvky v rámci scrollovacieho okna sa budú zobrazovať naľavo.
        self.scrollable_frame.setWidgetResizable(True)
        self.__graph_area_layout.setContentsMargins(0, 0, 0, 0)
        self.__graph_area_layout.setSpacing(0)
        self.__graph_area_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.__scrollbar_content.setLayout(self.__graph_area_layout)

        # Nastavenie obaľovacieho elementu pre obsah scrollovacieho okna. Bez tohto elementu by nebolo možné v rámci
        # scrollovacieho okna scrollovať.
        self.scrollable_frame.setWidget(self.__scrollbar_content)

        # Nastavenie fixnej šírky AddRemoveCombobox elementu a jeho následné do rozmiestnenia v rámci obsahu scrollova-
        # cieho okna. Fixná šírka bola zvolená z dizajnérskeho hľadiska.
        self.__add_rm_layer_rc.setFixedWidth(412)
        self.__graph_area_layout.addWidget(self.__add_rm_layer_rc)

    def initialize(self, logic_layer: GraphLogicLayer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inializuje atribúty triedy.
        Vytvorí predbežný zoznam názvov jednotlivých vrstiev. Po inicializácií AddRemoveComboboxFrame budú získane
        unikátne mená, ktoré sa použijú ako kľúče v slovníku.
        Na základe unikátnych mien je vytvorený slovník, ktorý prevádza meno vrstvy na jej poradové číslo a spätný
        slovník, ktorý prevádza poradové číslo na názov vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param logic_layer: referencia na logickú vrstvu aplikácie.
        """
        # Inicializácia panelu s možnosťami. Predovšetkým ide o skrytie tohto elementu a nastavenie aktívnej vrstvy na
        # None, ak ide už o opätovnú inicializáciu.
        self.__options_frame.initialize()

        # Nastavenie referencie na logickú vrstvu a vyčistenie slovníkov.
        self.__logic_layer = logic_layer
        self.__order_to_name_dict = {}
        self.__name_to_order_dict = {}

        # Priradí referenciu na list aktívnych vrstiev z logickej vrstvy.
        self.__active_layers = self.__logic_layer.active_layers

        # Priradí referenciu na list odkazov na triedy typu NeuralLayer. Taktiež uloží do premennej počet vrstiev
        # neurónovej siete.
        self.__neural_layers = self.__logic_layer.neural_layers
        self.__number_of_layers = len(self.__neural_layers)

        # Vytvorenie predbežného zoznamu názvov vrstiev na základe mien v keras modeli. Tento zoznam môže obsahovať
        # duplicity.
        layer_name_list = []
        for i in range(self.__number_of_layers):
            layer_name_list.append(self.__neural_layers[i].layer_name)

        # Inicializácia AddRemoveComboboxFrame. Metóda tejto triedy navracia list unikátnych názvov vrstiev.
        # Unikátne názvy su použité ako kľúče v slovníku a sú priradené príslušným vrstvám.
        unique_name_list = self.__add_rm_layer_rc.initialize(layer_name_list, self.show_layer,
                                                                 'Add layer', True, 'Select layer')
        # Vytvorenie slovníku s unikátnymi názvami vrstiev. Súčasne sú vrstvám priradené nové mená v prípade ak sa medzi
        # pôvodnými mentami objavili duplicity. Tieto mená totiž zohrávajú úlohu identifikátorov na pridávanie a
        # odoberanie vrstiev.
        for i, layerName in enumerate(unique_name_list):
            self.__neural_layers[i].layer_name = layerName
            self.__order_to_name_dict[i] = layerName
            self.__name_to_order_dict[layerName] = i

        # Ak je počet aktívnych vrstiev menší ako je počet všetkých vrstiev, je zobrazený AddRemoveComobobx, ktorý
        # umožňuje pridávanie vrstiev na základe ich mena. V opačnom prípade je tento element skrytý.
        if len(self.__active_layers) < self.__number_of_layers:
            self.__add_rm_layer_rc.show()
        else:
            self.__add_rm_layer_rc.hide()

        if len(self.__neural_layers):
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
        # Z premmennej si vyberieme názov vrstvy, ktorá bola zvolená. Na základe názvu získame index prislúchajúcej
        # vrstvy.
        layer_name = layer_tuple[1]
        layer_number = self.__name_to_order_dict[layer_name]

        # Otestuje, či je číslo vrstvy valídne.
        if 0 <= layer_number < self.__number_of_layers:
            layer_to_show = None

            # Inicializácia vrstvy na zobrazenie. V rámci nej sa na základe indexu vyberie vrstva neurónovej siete.
            # Pomocou metódy, sa vytvorí grafická reprezentácia vrstvy a odkaz na ňu je vrátený vo forme výstupnej
            # hodnoty metódy. Tejto metóde je nastavená minimálna šírka, kvôli tomu, aby sa postupným pridávaním
            # ďalších vrstiev zobrazil scrollbar. Bez nastavenia minimálnej šírky by po pridaní ďalších vrstiev
            # boli tieto elementy neprirodzene splošťované. Vytvorená grafická reprezentácia je nakoniec vlo-
            # žená za všetky predchádzajúce vrstvy a pred grafické vyobrazenie AddRemoveComobox elementu.
            if layer_number < self.__number_of_layers:
                layer_to_show = self.__neural_layers[layer_number]
                graph_frame_widget = layer_to_show.create_graph_frame(self.show_layer_options_frame, self.hide_layer)
                graph_frame_widget.setMinimumWidth(412)
                self.__graph_area_layout.insertWidget(self.__graph_area_layout.count() - 1, graph_frame_widget)

            # Poradové číslo vrstvy je následne vložené do listu aktívnych vrstiev. Tento list je použíaný pre efektí-
            # vnejšie vykonávanie funkcií ako sú update alebo prekreslenie, pretože niektoré zmeny sa prejavujú len
            # na nasledujúcich vrstvách a preto by bolo zbytočné vykonávať určité operácie pre vrstvy, ktoré ostanú
            # bez zmeny. Po pridaní vrstvy do listu aktívnych vrstiev je z AddRemoveComobox inštancie odobratá
            # položka s menom práve zobrazenej vrstvy.
            self.__active_layers.append(layer_number)
            self.__add_rm_layer_rc.hide_item(layer_name)

            # Novo pridanej vrstve sú priradené hodnoty výstupu na danej vrstve, ktoré má zobrazovať. Po priradení bodov
            # sú na vrstve aplikované zmeny, je použitá konfigurácia
            self.__logic_layer.set_points_for_layer(layer_number)
            layer_to_show.apply_displayed_data_changes()
            layer_to_show.use_config()
            layer_to_show.redraw_graph_if_active()

            # Ak je počet aktívnych vrstiev rovný celkovému počtu vrstiev je skrytý panel pre pridávanie nových vrstiev.
            if len(self.__active_layers) == self.__number_of_layers:
                self.__add_rm_layer_rc.hide()

    def hide_layer(self, layer_number: int):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje vrstvu, podľa jej poradového čísla.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorá má byť skrytá
        """
        # Najskôr sa skontroluje, či sa vrstva s daným poradovým číslom nachádza medzi aktívnymi vrstvami. Ak áno,
        # na základe poradového čísla sa nájde príslušné meno vrstvy a odkaz z listu odkazov na neurónové vrstvy.
        # Zobrazovaný graf pre danú vrstvu je spolu s ovládačom váh vyčistený a je mu priradená hodnota None,
        # aby zanikli referencie na tento element a garbage collector mohol uvolniť prostriedky používané
        # týmto elementom.
        if layer_number in self.__active_layers:
            layer_name = self.__order_to_name_dict[layer_number]
            layer = self.__neural_layers[layer_number]
            layer.graph_frame.clear()
            layer.graph_frame = None

            # Následne je číslo vrstvy odstránené z listu aktívnch vrstiev. A názov vrstvy je opätovne pridaný do
            # AddRemoveCombobox elementu, pre opätovné zobrazenie vrstvy. Ak je počet zobrazených vrstiev menší
            # ako počet všetkých vrstiev, je zobrazený AddRemoveCombobox element. Ak aktuálne skrytá vrstva
            # nastavená v rámci panelu možností, je tento panel skrytý.
            self.__active_layers.remove(layer_number)
            self.__add_rm_layer_rc.show_item(layer_name)
            if len(self.__active_layers) < self.__number_of_layers:
                self.__add_rm_layer_rc.show()
            if self.__options_frame.active_layer == layer:
                self.__options_frame.hide_option_bar()

    def show_layer_options_frame(self, layer_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazí na panele, možnosti zobrazenia pre požadovanú vrstvu.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param layer_number: číslo vrstvy, ktorej možnosti majú byť zobrazené.
        """
        # Inicializuje panel možnosti konfiguráciou vrstvy, ktorej čislo bolo poslané ako argument.
        if 0 <= layer_number < self.__number_of_layers:
            layer = self.__neural_layers[layer_number]
            self.__options_frame.initialize_with_layer_config(layer, layer.config)

    def apply_changes_on_options_frame(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností.
        """
        self.__options_frame.update_selected_config()

    def update_active_options_layer(self, start_layer):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností, ak je splnená podmienka

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: číslo vrstvy, na ktorej sa uskutočnila zmena.
        """
        self.__options_frame.update_active_options_layer(start_layer)

    def update_layer_if_active(self, neural_layer_ref):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aplikuje zmeny na panel možností, ak je argument rovný práve zobrazovanej vrstve na panely možností.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer_ref: referencia na vrstvu.
        """
        if neural_layer_ref == self.__options_frame.active_layer:
            self.apply_changes_on_options_frame()

    @property
    def logic_layer(self):
        return self.__logic_layer

    @logic_layer.setter
    def logic_layer(self, new_ref):
        self.__logic_layer = new_ref


class OptionsFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """"
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Obsahuje možnosti zobrazenia pre jednotlivé vrstvy. V rámci nej je možne navoliť zobrazované súradnice pre
        jednotlivé metódy, ofarbiť body na základe ich label, povoliť zobrazenie mriežky, ak je to možné,
        uzmaknutie pohľadu v grafe. Je možné aj zvoliť redukciu priestoru a zobraziť požadovaný
        PCA komponent alebo použiť metódu t-SNE.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__labels_entries_list:  list vstupnov pre zadávanie názvov osi v grafe aktívnej vrstvy.
        :var self.__cords_entries_list:   list vstupov na zadávanie zobrazovaných súradníc.
        :var self.__tSNE_parameters_dict: obsahuje grafické prvky pre zadávanie parametrov pre výpočet metódy t-SNE.
        :var self.__changed_config:       referencia na konfiguračný slovník vrstvy, pre ktorú sa zobrazujú možnosti.
        :var self.__active_layer:         odkaz na aktívnu vrstvu, pre ktorú sú aktuálne zobrazené možnosti.
        :var self.__layer_name_label:     zobrazuje meno aktívnej vrstvy, pre ktorú sú zobrazované možnosti.
        :var self.__bar_wrapper:          obaľovací element, obsahuje jednotlivé podskupiny možností.
        :var self.__choose_cords_gbox:    obaľovací element skupiny možností pre nastavenie zobrazovaných súradníc.
        :var self.__possible_cords_label: zobrazuje informáciu o rozsahu súradníc, ktoré môžu byť zobrazené.
        :var self.__choose_label_gbox:    obaľovací element pre skupinu možností zodpovednú za názvy osí grafu.
        :var self.__graph_view_gbox:      obaľovací element pre skupinu možností zodpovednú za možnosti vzhľadu a
                                          zobrazovaných prvkov vo vykresľovanom grafe.
        :var self.__color_labels_cb:      checkbox, ktorý zachytáva, či je požoadované aby v zobrazovanom grafe boli
                                          body ofarebené na základe príslušnosti k triede.
        :var self.__lock_view_cb:         checkbox, ktorý značí, či májú byť pri prekreslení grafu zanechaný posledne
                                          nastavený zoom, offset a prípadne rotácia.
        :var self.__3d_view_cb:           checbox, ktorý zachytáva, či majú byť vykreslené body v 2D alebo 3D grafe.
        :var self.__show_polygon:         checbox, ktorý značí, či majú byť vykreslené hrany polygónu priestoru, ktorý
                                          lemuje zadané body, ak je vstup neurónovej vrstvy 2D alebo 3D.
        :var self.__dim_reduction_gbox:   obaľovací element, zahŕňajúci možnosti pre redukciu priestoru.
        :var self.__actual_reduc_method:  label element, ktorý oboznamuje používateľa s aktuálne
                                          použitou metódou redkucie priestoru.
        :var self.__no_method_rb:         radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: No method
        :var self.__PCA_rb:               radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: PCA
        :var self.__t_SNE_rb:             radiobutton, ktorý sĺuži na nastavenie požadovanej metódy na redukciu
                                          priestoru, konkrétne použitie metódy: t-SNE
        :var self.__pca_info_gbox:        obaľovací element, ktorý obaĺuje prvky zobrazujúce informácie o vykonanej
                                          PCA analýze.
        :var self.__var_expl_lb:          listbox, ktorý zobrazuje informácie o vysvetlenej variabilite jednotlivými
                                          komponentami na zvolenej vrstve, ak bola použitá metóda PCA
        :var self.__load_scores_lb:       listbox, ktorý zobrazuje informácie o vpyve jednotlivých vstupov na zvolený
                                          hlavný komponent, ak bola použitá metóda PCA.
        :var self.__t_sne_parameter_gbox: obaľovací element, ktorý v sebe zahŕňa grafické prvky pre zadávanie parametrov
                                          použitých aplikovanie metódy t-SNE.
        :var self.__use_method_btn:       tlačidlo, na použitie zvolenej metódy pomocou radioboxu.
        :var self.__cur_used_method:      názov práve používanej metódy redukcie priestor.
        """
        super(OptionsFrame, self).__init__(*args, **kwargs)
        # Nastavenie veľkosti fontu pre celý panel.
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)

        # Definovanie základných atribútov.
        self.__labels_entries_list = []
        self.__cords_entries_list = []
        self.__tSNE_parameters_dict = {}
        self.__changed_config = None
        self.__active_layer = None

        # Definovanie grafických elementov.

        # Definícia obaľovacieho elementu pre všetky podskupiny možností a definovanie label elementu, ktorý bude
        # zobrazovať meno vrstvy, ktorej nastavenia môžeme meniť.
        self.__bar_wrapper = QGroupBox('Layer options')
        self.__layer_name_label = QLabel('Layer name')

        # Obaľovací element podskupiny zodpovednej za výber zobrazovaných súradníc. Taktiež je definovaný aj label
        # element, ktorý bude zobrazovať rozsah možných súradníc, ktoré sa dajú zobraziť.
        self.__choose_cords_gbox = QGroupBox('Displayed cords')
        self.__possible_cords_label = QLabel('Possible cords')

        # Obaľovací element pre podskupinu grafický prvkov starajúcich sa o zmeny názvu osí grafu.
        self.__choose_label_gbox = QGroupBox('Axis name')

        # Definíci elementu, ktorý obaľuje možnosti týkajúce sa vzhľadu grafu, zobrazovaného obsahu a pohľadu na graf.
        # Sú zadefinované checkboxy pre jednotlivé možnosti.
        self.__graph_view_gbox = QGroupBox('Graph view options')
        self.__color_labels_cb = QCheckBox('Color according to label')
        self.__lock_view_cb = QCheckBox('Lock view')
        self.__3d_view_cb = QCheckBox('3D view')
        self.__show_polygon_cb = QCheckBox('Show space grid')

        # Definícia obalu držiaceho grafické prvky, slúžiace na zmenu metódy redukcie priestoru. Taktiež sú zadefinované
        # aj grafické prvky, ktoré zobrazujú informácie o určitých parametroch métod, prípadne panel pre navolenie
        # parametrov potrebných pre použitie metódy.
        self.__dim_reduction_gbox = QGroupBox('Dimension reduction')
        self.__actual_reduc_method = QLabel('Actual used: No method')
        self.__no_method_rb = QRadioButton('No method')
        self.__PCA_rb = QRadioButton('PCA')
        self.__t_SNE_rb = QRadioButton('t-SNE')
        self.__pca_info_gbox = QGroupBox('PCA information')
        self.__var_expl_lb = QListWidget()
        self.__load_scores_lb = QListWidget()
        self.__t_sne_parameter_gbox = QGroupBox('t-SNE parameters')
        self.__use_method_btn = QPushButton()
        self.__cur_used_method = 'No method'

        # Grafická inicializácia panelu možností.
        self.initialize_ui()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia jednotlivých grafických prvkov.
        """
        # Vytvorenie inštancie rozmiestnenia, ktoré bude použité na rozmiestnenie jednotlivých podskupín elementov.
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(5, 0, 5, 5)
        main_layout.addWidget(self.__bar_wrapper)

        # Toto rozmiestnenie sa stará o podskupiny možností nastavení grafu. Rozmiestnenie je následne nastavené obalo-
        # vaciemu elementu.
        options_groups_layout = QVBoxLayout()
        options_groups_layout.setAlignment(QtCore.Qt.AlignTop)
        options_groups_layout.setSpacing(0)
        options_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.__bar_wrapper.setLayout(options_groups_layout)

        # Kvôli vizuálu je elementu, ktorý zobrazuje názov vrstvy pridaný margin.
        self.__layer_name_label.setMargin(10)

        # Vytvorenie fontu a následné priradenie fontu elementu, ktorý zobrazuje meno vrstvy. Font je nastavený na urči-
        # tú veľkosť a je aj nastavená možnosť hrubšieho písma. Po nastavení fontu je element vložený do rozmiestnenia.
        layer_name_font = QtGui.QFont()
        layer_name_font.setPointSize(10)
        layer_name_font.setBold(True)
        self.__layer_name_label.setFont(layer_name_font)
        options_groups_layout.addWidget(self.__layer_name_label, alignment=QtCore.Qt.AlignCenter)

        # Vytvorenie rozmiestnenia pre kontajner držiaci grafické elementy pre zmenu zobrazovaných súradníc a jeho ná-
        # sledné nastavenie tomuto kontajneru. Následne je do tohto rozmiestnenia vložený element na zobrazovanie
        # rozsahu súradníc, ktoré je možné zobraziť. Taktiež je tento kontajner vložený do rozmiestnenia
        # pre podskupiny nastavení.
        choose_cords_layout = QVBoxLayout()
        choose_cords_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__choose_cords_gbox.setLayout(choose_cords_layout)
        choose_cords_layout.addWidget(self.__possible_cords_label, alignment=QtCore.Qt.AlignHCenter)
        options_groups_layout.addWidget(self.__choose_cords_gbox)

        self.__possible_cords_label.setContentsMargins(0, 0, 0, 0)

        # Do rozmiestnenia pre podskupiny je vložený obaľovací element pre podskupinu prvkov zodpovedných za zmenu ozna-
        # čení jednotlivých osí. Následje je vytvorená inštancia rozmiestnenia pre tento obaľovací element a v rámci
        # rozmiestnenia je nastavené zarovnávanie naľavo.
        options_groups_layout.addWidget(self.__choose_label_gbox)
        choose_labels_layout = QVBoxLayout()
        self.__choose_label_gbox.setLayout(choose_labels_layout)
        choose_labels_layout.setAlignment(QtCore.Qt.AlignTop)

        # V nasledujúcom cykle sú vytvorené a inicializované grafické prvky pre vstup údajov od používateľa. Ide o vstu-
        # pné prvky pre zadávanie požadovaných zobrazovaných súradníc a vstupy na zadávanie označení jednotlivých osí.
        # Ide o grafické prvky typu RewritableLabel, ktorým je nastavený identifikátor, názov, preddefinovaná hodnota
        # a funkcia, ktorá sa zavolá pri aktivácií prvku. Prvky sú postupne pridávané do rozmiestení, ku ktorým
        # patria.
        label_entry_names = ['X axis label:', 'Y axis label:', 'Z axis label:']
        label_entry_values = ['Label X', 'Label Y', 'Label Z']
        cords_entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
        for i in range(3):
            rw_label = RewritableLabel(i, cords_entry_names[i], str('-'), self.validate_cord_entry)
            choose_cords_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__cords_entries_list.append(rw_label)
            rw_label = RewritableLabel(i, label_entry_names[i], label_entry_values[i], self.validate_label_entry)
            choose_labels_layout.addWidget(rw_label, alignment=QtCore.Qt.AlignLeft)
            self.__labels_entries_list.append(rw_label)

        # Do rozmiestnenia podskupín možností je pridaná podskupina zaoberajúca sa vzhľadom a zobrazovaným obsahom gra-
        # fu. Je vytvorené ďalšie rozmiestnenie, ktoré je nastavené obaľovaciemu elementu tejto podskupiny.
        # V rámci tohto rozmiestnenia budú grafické elementy zarovnávané naľavo.
        options_groups_layout.addWidget(self.__graph_view_gbox)
        view_options_layout = QVBoxLayout()
        self.__graph_view_gbox.setLayout(view_options_layout)
        view_options_layout.setAlignment(QtCore.Qt.AlignTop)

        # Checboxom pre jednotlivé možnosti je nastavená funkcia, ktorá sa zavolá po kliknutí na možnosť. Po nastavení
        # sú jednotlivé checkboxy pridané do rozmiestnenia.
        self.__color_labels_cb.toggled.connect(self.on_color_label_check)
        self.__lock_view_cb.toggled.connect(self.on_lock_view_check)
        self.__3d_view_cb.toggled.connect(self.on_3d_graph_check)
        self.__show_polygon_cb.toggled.connect(self.on_show_polygon_check)
        view_options_layout.addWidget(self.__color_labels_cb)
        view_options_layout.addWidget(self.__lock_view_cb)
        view_options_layout.addWidget(self.__3d_view_cb)
        view_options_layout.addWidget(self.__show_polygon_cb)

        # Do rozmiestnenia pre podskupiny možností je pridaná podskupina obsahujúca možnosti na redukciu priestoru.
        # Je vytvorená ďalšia inštancia vertikálneho rozmiestnenia, ktoré je nastavené tomuto elementu.
        options_groups_layout.addWidget(self.__dim_reduction_gbox)
        dim_reduction_layout = QVBoxLayout()
        dim_reduction_layout.setContentsMargins(0, 0, 0, 0)
        dim_reduction_layout.setAlignment(QtCore.Qt.AlignTop)
        self.__dim_reduction_gbox.setLayout(dim_reduction_layout)

        # Elementu, ktorý informuje o práve používanej metóde redukcie priestoru je nastavený margin a je element je
        # pridaný do rozmiestnenia obaľovacieho elementu. Súčasne mu je nastavené aj zarovnanie.
        self.__actual_reduc_method.setMargin(10)
        dim_reduction_layout.addWidget(self.__actual_reduc_method,
                                       alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)

        # Vytvorenie obaľovacieho elementu, ktorý zaručí aby pri zakliknutí nejakého radio buttonu boli ostatne odzna-
        # čené. Taktiež je vytvorená rozmiestnenie, do ktorého budú vkladané jednotlivé radio buttony na voľbu metódy
        # použitej na redukciu priestoru. Do rozmiestnenia podskupiny na redukciu priestoru je pridané rozmiestnenie
        # pre radio buttony. Následne je radio buttononom nastavená funkcia, ktorá sa má spustiť po kliknutí. Na
        # záver je nastavené zakliknutie na radio buttone pre možnosť No method.
        radio_button_group = QButtonGroup()
        radio_button_layout = QHBoxLayout()
        radio_button_layout.setContentsMargins(5, 0, 5, 5)
        radio_button_layout.setAlignment(QtCore.Qt.AlignLeft)
        dim_reduction_layout.addLayout(radio_button_layout)
        self.__no_method_rb.toggled.connect(self.on_method_change)
        self.__PCA_rb.toggled.connect(self.on_method_change)
        self.__t_SNE_rb.toggled.connect(self.on_method_change)
        self.__no_method_rb.setChecked(True)

        # Radio buttony sú popridávané do skupiny tlačidiel a taktieź aj do rozmiestnenia v rámci obaľovacieho elementu
        # týchto tlačidiel.
        radio_button_group.addButton(self.__no_method_rb)
        radio_button_layout.addWidget(self.__no_method_rb)
        radio_button_group.addButton(self.__PCA_rb)
        radio_button_layout.addWidget(self.__PCA_rb)
        radio_button_group.addButton(self.__t_SNE_rb)
        radio_button_layout.addWidget(self.__t_SNE_rb)

        # Listboxu zobrazujúcemu variabilitu vysvetlenú jednotlivými komponentami je priradená funkcia, ktorá sa zavolá
        # po kliknutí na komponent. Obaľovaciemu elementu listboxov je nastavená maximálna výška. Listbox je pridaný
        # do rozmiestnenia obaľovacieho elementu pre tento listbox so zarovnaním nahor, toto zarovnanie spôsobí, že
        # sa natiahne po maximálnu výšku obaľovacieho elementu.
        self.__var_expl_lb.currentRowChanged.connect(self.show_PCA_loading_scores)
        self.__pca_info_gbox.setMaximumHeight(200)
        dim_reduction_layout.addWidget(self.__pca_info_gbox, alignment=QtCore.Qt.AlignTop)

        # Vytvorenie rozmiestnenia pre podskupinu obsahujúcu informácie o výstupe metódy PCA na danej vrstve.
        pca_info_layout = QHBoxLayout(self.__pca_info_gbox)
        pca_info_layout.setAlignment(QtCore.Qt.AlignTop)
        pca_info_layout.setContentsMargins(0, 10, 0, 0)

        # Vytvorenie obaľovacieho elementu pre listbox vysvetľujúci variabilitu a pridanie tohto kontajnera do
        # rozmiestnenia informačnej podskupiny.
        var_expl_gbox = QGroupBox('PC variance explanation')
        pca_info_layout.addWidget(var_expl_gbox)

        # Vytvorenie rozmiestnenia pre obaľovací element listboxu zobrazujúceho vysvetlenú variabilitu.
        layout_var_expl = QVBoxLayout(var_expl_gbox)
        layout_var_expl.setAlignment(QtCore.Qt.AlignTop)
        layout_var_expl.setContentsMargins(0, 0, 0, 0)

        # Listboxu vysvetľujúcej variability pre jednotlivé komponenty je nastavená maximálna šírka a listbox je
        # pridaný do rozmiestnenia.
        self.__var_expl_lb.setMaximumWidth(150)
        layout_var_expl.addWidget(self.__var_expl_lb)

        # Vytvorenie obaľovacieho elementu pre listbox, pojednávajúci o vplyve vstupov na zvolený hlavný komponent.
        loading_score_gbox = QGroupBox('Loading scores')
        pca_info_layout.addWidget(loading_score_gbox)

        # Vytvorenie rozmiestnenia pre obaľovací element, listboxu s hodnotami vpyvu vstupov na jednotlivé komponenty.
        # Následne je listbox pridaný do rozmiestnenia.
        loading_score_layout = QVBoxLayout(loading_score_gbox)
        loading_score_layout.setAlignment(QtCore.Qt.AlignTop)
        loading_score_layout.setContentsMargins(0, 0, 0, 0)
        self.__load_scores_lb.setMaximumWidth(150)
        loading_score_layout.addWidget(self.__load_scores_lb)

        # Pridanie obaľovacieho elementu pre vsupné prvky na definíciu parametrov metódy t-SNE do rozmiestnenia podsku-
        # piny možností pre redukciu priestoru.
        dim_reduction_layout.addWidget(self.__t_sne_parameter_gbox, alignment=QtCore.Qt.AlignTop)
        t_sne_par_layout = QVBoxLayout(self.__t_sne_parameter_gbox)
        t_sne_par_layout.setAlignment(QtCore.Qt.AlignTop)

        # Vytvorenie listov obsahujúcich identifikátor a názov pre inštancie triedy RewritableLabel. Identifikátor je
        # zhodný s kľúčom do konfiguračnej premennej pre výpočet metódy t-SNE.
        t_sne_parameter_id_list = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']
        t_sne_parameter_label = ['Number of components:', 'Perplexity:', 'Early exaggeration:', 'Learning rate:',
                                 'Number of iteration:']
        for i in range(len(t_sne_parameter_id_list)):
            t_sne_parameter_rw = RewritableLabel(t_sne_parameter_id_list[i], t_sne_parameter_label[i], '-',
                                                 self.validate_t_sne_entry)
            t_sne_par_layout.addWidget(t_sne_parameter_rw, alignment=QtCore.Qt.AlignLeft)
            self.__tSNE_parameters_dict[t_sne_parameter_id_list[i]] = t_sne_parameter_rw

        # Nastavenie textu tlačidla, jeho pridanie do rozmiestnenia a nastavenie funkcie, ktorá bude zavolaná po kliknu-
        # tí na tlačidlo.
        self.__use_method_btn.setText('Use method')
        dim_reduction_layout.addWidget(self.__use_method_btn)
        self.__use_method_btn.clicked.connect(self.use_selected_method)

    def initialize(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vyčistenie atribútov a skrytie celého options baru.
        """
        self.__changed_config = None
        self.__active_layer = None
        self.hide_option_bar()

    def initialize_with_layer_config(self, neural_layer: NeuralLayer, config: dict):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie aktuálnej aktívnej vrstvy a configu tejto vrstvy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param neural_layer: odkaz na aktívnu vrstvu, pre ktorú sú menené nastavenia a ktoré budú následne na túto
                             vrstvu použité
        :param config: odkaz na config aktuálne zobrazovanej a upravovanej vrstvy
        """
        self.__active_layer = neural_layer
        self.__changed_config = config
        self.update_selected_config()

    def initialize_label_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie názvov jednotlivých vstupov pre časť s označením osi grafov na základe načítaného configu.
        """
        # Zo zvoleného konfigu je vybratý parameter pojednávajúci o maximálnej možnej zobrazenej dimenzií. Tá môže byť
        # od 1 až po 3. Na základe tohto údaja bude zobrazený zodpovedajúci počet vstupov pre označenie osí grafov.
        # Z listu grafov sú postupne vyberané odkazy na grafické vstupy a je nastavená ich hodnota na základe
        # základných alebo posledne nastavených názvov osí. Element je následne zobrazený.
        number_of_possible_dim = self.__changed_config['max_visible_dim']
        for i in range(number_of_possible_dim):
            label_entry = self.__labels_entries_list[i]
            label_entry.set_variable_label(self.__changed_config['axis_labels'][i])
            label_entry.show()

    def initialize_view_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie hodnôt jednotlivých vstupov pre časť s možnosťami zobrazovania grafu na základe načítaného configu.
        """
        # Na základe pravdivostnej hodnoty sú začiarknuté alebo odznačené jednotlivé môžnosti pre vzhľad grafu.
        self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
        self.__lock_view_cb.setChecked(self.__changed_config['locked_view'])

        # Z konfiguračnej premennej je zistený maximálny počet viditeľných dimenzií. Na základe tohto údaju su niektoré
        # checkboxy zobrazené alebo ponechané naďalej skryté.
        number_of_possible_dim = self.__changed_config['max_visible_dim']
        if number_of_possible_dim >= 3:
            self.__3d_view_cb.setChecked(self.__changed_config['draw_3d'])
            self.__3d_view_cb.show()

        # Podľa toho či je vstup do siete 2D alebo 3D je možné zobraziť mriežku ohraničujúcu body. Ak nie táto možnosť
        # nie je príspustná, checkbox ostane skrytý. Ak je táto možnosť prípustná, je nastavená posledne zvolená
        # možnosť, ktorú používateľ zadal.
        if self.__changed_config['possible_polygon']:
            self.__show_polygon_cb.setChecked(self.__changed_config['show_polygon'])
            self.__show_polygon_cb.show()

        # Ak boli pre body načítané labels, naskytne sa možnosť ofarbiť body na základe príslušnosti k triede. Ak žiadne
        # labels neboli načítané, zostane táto možnosť vypnutá a naďalej skrytá.
        if self.__active_layer.possible_color_labels:
            self.__color_labels_cb.setChecked(self.__changed_config['color_labels'])
            self.__color_labels_cb.show()

    def initialize_dimension_reduction_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie hodnôt jednotlivých vstupov pre časť s možnosťami pre redukciu priestoru.
        """
        # Je nastavený názov aktuálne použitej metódy. A do premennej je vložený názov posledne zvolenej metódy. Metóda
        # nemusí byť ani zvolená. Budú nastavené predbežné nastavenia pre túto metódu.
        self.set_actual_method_lable(self.__cur_used_method)
        config_selected_method = self.__changed_config['config_selected_method']

        # Všetky radio buttony zodpovedajúce metódam sú zbavené zakliknutia.
        self.__no_method_rb.setChecked(False)
        self.__PCA_rb.setChecked(False)
        self.__t_SNE_rb.setChecked(False)

        # Podľa posledne zakliknutej metódy je zvolený príslušný radio button.
        if config_selected_method == 'No method':
            self.__no_method_rb.setChecked(True)
        elif config_selected_method == 'PCA':
            self.__PCA_rb.setChecked(True)
        elif config_selected_method == 't-SNE':
            self.__t_SNE_rb.setChecked(True)

        # Ak je aktuálne použitá metóda PCA, sú vypísané aj informácie o výstupe metódy.
        if self.__cur_used_method == 'PCA':
            self.update_PCA_information()

        # Následne sú prednastavené hodnoty parametrov pre t-SNE pre danú vrstvu.
        self.initialize_t_sne_parameters()
        self.on_method_change()

    def initialize_t_sne_parameters(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Predvyplnenie parametrov pre metódu t-SNE.
        """
        # Z konfiguračnej premennej pre aktívnu vrstvu je vybraná podzložka pre s-SNE konfiguráciu. Z tejto konfigurácie
        # sú získané hodnoty pre aktuálne použité parametre a pre posledne nastavené parametre.
        t_sne_config = self.__changed_config['t_SNE_config']
        actual_used_config = t_sne_config['used_config']
        options_config = t_sne_config['options_config']

        # Z konfigurácie pre hraničné hodnoty parametrov je vybraný maximálny počet možných komponentov. Na základe
        # tohoto čísla je nastavený názov vstupného grafického prvku, aby bol používateľ informovaný o tom, koľko
        # komponentov môže maximálne použiť.
        number_of_components = t_sne_config['parameter_borders']['n_components'][2]
        self.__tSNE_parameters_dict['n_components'].set_label_name(
            f'Number of components (max {number_of_components}):')

        # Následne sú v cykle na základe kľúčov prednastavené hodnoty vstupov.
        for key in self.__tSNE_parameters_dict:
            rewritable_label = self.__tSNE_parameters_dict[key]
            rewritable_label.show_variable_label()
            rewritable_label.set_variable_label(options_config[key])

            # Ak sa aktuálne používaná hodnota parametra nerovná naposledy nastavenej hodnote parametra je tento
            # parameter označený ako zmenený no ešte nepoužitý. Vyznačuje sa to tým, že je názov vyznačený
            # červenou farbou a je pridaná hviezdička.
            if actual_used_config[key] == options_config[key]:
                rewritable_label.set_mark_changed(False)
            else:
                rewritable_label.set_mark_changed(True)

    def use_selected_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Funkcia, ktorá sa zavolá po zvolení metódy a klikutí na tlačidlo Use method. Na základe zvolenej metódy, nastaví
        v konfiguračnej premennej hodnotu aktuálnej metódy a taktiež prispôsobí vzhľad a hodnoty na panele možností.
        """
        if self.__active_layer is not None:
            # Je získaná metóda, podľa zakliknutého radio buttonu. Je definovaná premenná need_recalculation, aby
            # nedošlo k zbytočnému prepočítavaniu na vrstve, nedošlo k žiadne zmene ale napríklad bolo tlačidlo
            # viac krát stlačené.
            method = self.get_checked_method()
            need_recalculation = False

            # Ak je začiarknutá metóda t-SNE, zistí sa či boli zmenené hodnoty niektorých parametrov oproti naposledy
            # použitým. Ak áno, do premennej need_recalculation sa priradí hodnota True ak nie, priradí sa False.
            if method == 't-SNE':
                need_recalculation = self.apply_t_SNE_options_if_changed()

            # Ak sa zakliknutá metóda líši od aktuálne používanej metódy.
            if method != self.__changed_config['used_method']:
                # Ak je zvolená metóda PCA, je potrbné nastaviť zobrazované súradnice aby nedošlo ku chybe. Zoznam súra-
                # dníc získame na základe minimálnej hodnoty spomedzi výstupnej dimenzie, pretože nie je možné zobrazo-
                # vať viac hlavných komponentov ako je počet výstupných hodnôt, počtu vzoriek, pretože PCA vytvorí len
                # minimum(počtu výstupných príznakov, počet vzoriek) hlavných komponentov a čísla 3, pretože je možné
                # zobraziť maximálne 3 dimenzie. Na základe tohto čísa sa vytvorí list súradníc, ktorý je priadený
                # do konfiguračnej premennej. Ďalej sa nastaví premenná need_recalculation na True, pretože zmena
                # metódy zaručuje potrebu vykonania prepočtu a nastavenia súradníc. Na záver je priradená do
                # konfiguračného súboru priradená zvolená metóda a aj v rámci panelu možností je nastavená
                # hodnota pre premennú self.__cur_used_method značiaca práve používanú metódu.
                if method == 'PCA':
                    self.__active_layer.initialize_default_PCA_cords()

                need_recalculation = True
                self.__changed_config['used_method'] = self.__cur_used_method = method
            # V prípade ak došlo k zmenám je potrebné aplikovať zmeny a prekresliť graf. Ak nedošlo k zmene, metóda kon-
            # čí. Podľa použitej metódy sú zobrazené a aktualizované aj dodatočné údaje.
            if need_recalculation:
                self.__changed_config['apply_changes'] = True
                self.__active_layer.use_config()
                self.__active_layer.redraw_graph_if_active()
                self.set_actual_method_lable(method)

                # Hide all informations
                self.hide_all_methods_information()
                if method == 'PCA':
                    self.update_PCA_information()
                    self.__pca_info_gbox.show()
                elif method == 't-SNE':
                    self.__t_sne_parameter_gbox.show()

                self.set_actual_method_lable(method)
            self.set_cords_entries_according_chosen_method()

    def update_selected_config(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Aktualizovanie jednotlivých možností na základe configu aktívnej vrstvy.
        """
        # Podľa aktívneho konfigu sú nastavené hodnoty grafických prvkov. Je nastavené meno vrstvy, sú skryté všetky
        # časti panelu, ktorých zobrazenie závisí od určitých podmienok. Z konfiguračného súboru je priradené metó-
        # da, používana na redukciu priestoru.
        if self.__active_layer is not None and self.__changed_config is not None:
            self.__cur_used_method = self.__changed_config['used_method']
            self.hide_option_bar()
            self.__bar_wrapper.show()
            self.__layer_name_label.setText(str(self.__changed_config['layer_name']))

            # Inicializácia grafických komponentov jednotlivých podskupín v závislosti od aktívnej konfiguračnej premen-
            # nej.
            self.set_cords_entries_according_chosen_method()
            self.initialize_label_options()
            self.initialize_view_options()
            self.initialize_dimension_reduction_options()

    def update_PCA_information(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vypísanie aktuálnych informácii o PCA.
        """
        # Listboxy držiace informačné hodnoty sú vyčistené a z konfiguračnej premennej je vybraný séria, ktorá obsahuje
        # informácie o tom, akú variabilitu vysvetľujú jednotlivé hlavné komponenty.
        self.__var_expl_lb.clear()
        self.__load_scores_lb.clear()
        variance_series = self.__changed_config['PCA_config']['percentage_variance']
        if variance_series is not None:
            # Do premennej pc_labels je priradený index série, zodpovedajúci názvu kompnentu v tvare PC{poradové číslo}.
            # Tieto hodnoty sú postupne vkladané do listboxu, spolu s príslušnou percentuálnou hodnotou variability,
            # ktorú daný komponent vyjadruje.
            pc_labels = variance_series.index
            for i, label in enumerate(pc_labels):
                self.__var_expl_lb.insertItem(i, '{}: {:.2f}%'.format(label, round(variance_series[label], 2)))

            # V informačnom listboxe, ktorý nesie informáciu o vplyve jednotlivých výstupov na komponenty sú zobrazené
            # hodnoty pre prvý hlavný komponent.
            self.show_PCA_loading_scores(0)

    def show_PCA_loading_scores(self, component_number):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Na základe parametra component_number vloží do listboxu informácie o vplyve výstupov na hlavný komponent,
        ktorého číslo bolo zadané ako argument.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param component_number: poradové číslo komponentu, o ktorom majú byť zobrazené informácie
        """
        # Testuje sa, či bolo zadané valídne číslo komponentu. Následne je listbox vyčistený, ak náhodou obsahoval staré
        # hodnoty. Následne je z konfiguračnej premennej na zákalde čísla komponentu o ktorom požadujeme informácie z
        # premennej typu dataframe získaný riadok, obsahujúci vplyv výstupov na požadovaný hlavný komponent.
        if component_number > -1:
            self.__load_scores_lb.clear()
            loading_scores = self.__changed_config['PCA_config']['largest_influence'][component_number]

            # Hodnoty sú zoradené podľa vpyvu zostupne, čiže na vrchu je uvedený výstup, ktorý ma daný komponent najvä-
            # čší vplyv. Zo zoradeného dataframe sa vyberú zoradené indexy. Tie sú v cykle vypísané spoločne s mierou
            # ich vplyvu na hlavný komponent.
            sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
            sorted_indexes = sorted_loading_scores.index.values
            for i, label in enumerate(sorted_indexes):
                self.__load_scores_lb.insertItem(i, '{}: {:.4f}'.format(label, round(loading_scores[label], 4)))

    def update_active_options_layer(self, start_layer=-1):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zobrazenie aktuálnych informácií pre aktívnu vrstvu, ktorej možnosti sú zobrazované.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param start_layer: poradové číslo najnižšej vrstvy, pri ktorej došlo ku zmene, ak je číslo menšie ako poradové
                            číslo aktívnej vrstvy, ktorej možnosti sú zobrazované, sú tieto možnosti aktualizované,
                            pretože mohlo dôjsť k zmenám v možných parametroch, prípadne k zmene hodnôt v rámci PCA.
        """
        # Pre istotu sa skontroluje, či je aktívna nejaká vrstva a tak isto, či je prítomná konfiguračná premenná pre
        # danú vrstvu. Následne sa zistí použitá metóda. Ak ide o metódu PCA, mohlo dôjsť k zmenám vo vyjadrenej var-
        # iabilite, prípadne vplyvu výstupu na hlavný komponent. K tomu mohlo dôjsť ak došlo k zmene na v poradí
        # nižšej alebo rovnakej vrstve.
        if self.__active_layer is not None and self.__changed_config is not None:
            actual_method = self.__changed_config['used_method']
            if actual_method == 'PCA' and start_layer <= self.__active_layer.layer_number:
                self.update_PCA_information()

    def apply_t_SNE_options_if_changed(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda priradí do konfiguračnej premennej hodnoty nastavených parametrov a testuje, či došlo k zmene oproti
        posledne nastaveným hodnotám. Ak došlo k zmene aspoň pri jednom parametre je premennej changed nastavená
        hodnota True. Výsledná hodnota tejto premennej je návratovou hodnotou metódy.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: vracia hodnotu
        """
        changed = False
        if self.__changed_config is not None:
            # Z konfiguračnej premennej získame konfiguráciu pre metódu t-SNE. Do premenných si uložíme aktuálne použí-
            # vanú a posledne použitú konfiguráciu. Následne tento prechádzame tieto konfigurácie v cykle a zisťujeme
            # či sa líšia. Ak áno, nastaví sa nová hodnota parametra ako používaná hodnota parametra a premenná
            # changed zmení svoju hodnotu na True, čo symbolizuje, že došlo k zmene.
            t_sne_config = self.__changed_config['t_SNE_config']
            used_config = t_sne_config['used_config']
            options_config = t_sne_config['options_config']
            for key in used_config:
                if used_config[key] != options_config[key]:
                    changed = True
                    used_config[key] = options_config[key]
            # Ak došlo k zmene, je potrebné na základe parametra pre metódu t-SNE nastaviť súradnice, ktoré sa majú
            # zobraziť na jednotlivých osiach.
            if changed:
                self.__active_layer.initialize_default_tSNE_cords()
                self.set_entries_not_marked(self.__tSNE_parameters_dict.values())
        return changed

    def set_cords_entries_according_chosen_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie textu a používaných hodnôt na základe zvolenej metódy.
        :return:
        """
        # Na základe aktuálne použitej metódy an redukciu priestoru sú definované hodnoty, ktoré sa majú nastaviť a
        # zobraziť na paneli možností v jednotlivých podskupinách.
        if self.__cur_used_method == 'No method':

            # Ak je aktuálne zvolená metóda No method pokračuje sa touto vetvou. Následne sa nastavia hodnoty pre
            # premenné, podľa toho, či sú daná vrstva obsahuje feature mapy alebo nie.
            if self.__changed_config['has_feature_maps']:

                # Ak aktívna vrstva obsahuje feature mapy, sú názvy a hodnoty vstupov preddefinované inak, pretože
                # mapy sú dvojrozmerné a pre lepšiu predstavu a pohdlnejšie zobrazovanie je vhodné použiť systém
                # zobrazovania (riadok, stĺpec). Informácia o prípustných súradniciach je vložená do premennej
                # cords_label text. Umožňuje zobraziť súradnice od 0 po prvé číslo mínus 1 pre voľbu riadku
                # a srúadnicu od 0 po druhé číslo mínus 1 pre voľbu stĺpca.
                output_shape = self.__changed_config['output_shape']
                cords_label_text = 'Possible cords: {}x{}'.format(output_shape[-3], output_shape[-2])
                tmp_displayed_cords_tuples = list(zip(self.__changed_config['no_method_config']['displayed_cords'][0],
                                                      self.__changed_config['no_method_config']['displayed_cords'][1]))
                # Na základe naposledy nastanených zobrazovaných súradníc je naplnená premenná displayed cords.
                displayed_cords = []
                for x, y in tmp_displayed_cords_tuples:
                    displayed_cords.append(f'{x}, {y}')
            else:
                # Ak vrstva nebsahuje feature mapy, je informatívny text nastavený na prípustné zobrazené hodnoty.
                cords_label_text = 'Possible cords: 0-{}'.format(self.__changed_config['output_dimension'] - 1)
                displayed_cords = self.__changed_config['no_method_config']['displayed_cords']
            entry_names = ['Axis X:', 'Axis Y:', 'Axis Z:']
            possible_cords = self.__changed_config['max_visible_dim']

        elif self.__cur_used_method == 'PCA':

            # Ak je aktuálne používaná metóda PCA, sú nastavené názvy vstupov a hodnoty vstupov sú nastavené na zákaldne
            # posledne použitých hodnôt zadaných používateľom.
            entry_names = ['PC axis X:', 'PC axis Y:', 'PC axis Z:']
            number_of_pcs = min(self.__changed_config['output_dimension'],
                                self.__changed_config['number_of_samples'])
            # Podľa počtu hlavných komponentov, ktoré je možné zobraziť je do premennej obsahujúcej informáciu o rozsahu
            # možných zadaných hodnôt priradená informácia.
            if number_of_pcs == 0:
                # Ak je počet komponentov 0, znamená to, že nie je možné zobraziť žiadny komponent, preto je do
                # premennej táto informácia priradená.
                cords_label_text = 'No possible PCs:'
            else:

                # Ak je počet komponentov rozdielny od nuly, je do premennej priradená informácia o tom, že je možné
                # voliť z rozsahu 1 až po číslo udávajúce počet komponentov.
                cords_label_text = 'Possible PCs: 1-{}'.format(number_of_pcs)

            # Hodnoty následných troch premenných nazávisia od toho, či vrstva obsahuje alebo neobsahuje feature mapy.
            possible_cords = min(number_of_pcs, self.__changed_config['max_visible_dim'])
            displayed_cords = self.__changed_config['PCA_config']['displayed_cords'].copy()
            displayed_cords = np.array(displayed_cords) + 1
        elif self.__cur_used_method == 't-SNE':

            # Podobne ako pri ostatných metódach je sú aj pre metódu t-SNE preddefinované názvy a hodnoty, ktoré sa majú
            # nastaviť vstupom.
            entry_names = ['t-SNE X:', 't-SNE Y:', 't-SNE Z:']
            possible_cords = self.__changed_config['t_SNE_config']['used_config']['n_components']
            cords_label_text = 'Possible t-SNE components: 0-{}'.format(possible_cords - 1)
            displayed_cords = self.__changed_config['t_SNE_config']['displayed_cords'].copy()

        # Inicializované premenné sú následne zaslané ako argumenty funkcie, ktorá sa stará o zobrazenie názvova hodnôt
        # grafických vstupov pre zadávanie zobrazených súradníc.
        self.set_cords_entries(entry_names, cords_label_text, displayed_cords, possible_cords)

    def set_cords_entries(self, entry_names, cords_label_text, displayed_cords, possible_cords):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví názvy a hodnoty vstupov na základe poskytnutých argumentov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param entry_names:       predstavuje list názvov pre jednotlivé grafické vstuy súradníc.
        :param cords_label_text: obsahuje informáciu o tom, z akého rozashu je možné zvoliť súradnice.
        :param displayed_cords:  list nastavených zobrazovaných hodnôt, ktoré majú byť nastavené ako preddefinované.
        :param possible_cords:   predstavuje počet súradníc, ktoré je možné zobraziť.
        """
        # Najskôr sú všetky súradnicové vstupy skryté. Následne je nastavený text informačnému elementu. Potom v cykle
        # v závislosti od počtu možných súradníc sú zobrazované vstupy a sú im nastavené hodnoty na základe zaslaných
        # argumentov.
        self.hide_entries(self.__cords_entries_list)
        self.__possible_cords_label.setText(str(cords_label_text))
        for i in range(possible_cords):
            cord_entry_rewritable_label = self.__cords_entries_list[i]
            cord_entry_rewritable_label.set_label_name(entry_names[i])
            cord_entry_rewritable_label.set_variable_label(displayed_cords[i])
            cord_entry_rewritable_label.show()

    def set_actual_method_lable(self, method_name):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda nastaví text grafickému prvku, ktorý informuje používateľa o aktuálne používanej metóde
        redukcie priestoru.

        Parameter
        ----------------------------------------------------------------------------------------------------------------
        :param method_name: meno aktuálne zvolenej metódy
        """
        self.__actual_reduc_method.setText(str(f'Actual used: {method_name}'))

    def set_entries_not_marked(self, entries_list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda označí všetky prvky typu RewritableLabel ako neopozmené, čiže im nastaví defaultnu čiernu farbu a
        odstráni hviezdičku.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param entries_list: list obsahujúci odkazy na RewritableLabel, ktoré treba označiť ako nezmenené.
        """
        for entry in entries_list:
            entry.set_mark_changed(False)

    def hide_option_bar(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na skrytie obsahu panelu možností. Skryje obaľovací element a následne aj jednotlivé podskupiny.
        :return:
        """
        self.__bar_wrapper.hide()
        self.hide_option_bar_items()

    def hide_option_bar_items(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda ktorá postupne vola funkie na skrytie grafických prvkov v jednotlivých podskupinách.
        """
        self.hide_entries(self.__labels_entries_list)
        self.hide_entries(self.__cords_entries_list)
        self.hide_graph_view_options()
        self.hide_dimension_reduction_options()

    def hide_entries(self, list_of_entries):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda skryje zadanú množinu vstupných grafických prvkov typu RewritableLabel.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param list_of_entries: list odkazov typu RewritableLabel, ktoré majú byť skryté.
        """
        for entry in list_of_entries:
            entry.show_variable_label()
            entry.hide()

    def hide_graph_view_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Skryje grafické prvky týkajúce sa zobrazenia a obsahu grafu. Sú skryté len prvky, pri ktorých sa môže stať,
        že nebudú zobrazené. Napríklad checkbox lockview je zobrazený v každom prípade, no chebox color label
        sa zobrazí len v prípade ak boli nahrané labels pre načítané body.
        """
        self.__color_labels_cb.hide()
        self.__3d_view_cb.hide()
        self.__show_polygon_cb.hide()

    def hide_dimension_reduction_options(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na skrytie grafických prvkov v rámci podskupiny redukcia priestoru.
        """
        self.hide_all_methods_information()

    def hide_all_methods_information(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda skryje obaľovacie elementy, ktoré sú v súčasťou podskupiny ohľadom redukcie priestoru.
        """
        self.__pca_info_gbox.hide()
        self.__t_sne_parameter_gbox.hide()

    def on_method_change(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa zavolá po kliknutí na radio button. V rámci tejto metódy sú pre zvolenú metódu zobrazené prípadné
        dodatočné okná.
        """
        # Pre istotu sa otestuje, či je inicializovaná konfiguračná premenná.
        if self.__changed_config is not None:
            # Je získaný názov metódy, ktorá bola zvolená. Do konfiguračnej premennej je zapísaná hodnota zakliknutej
            # metódy pre udržanie informácie o tom, na akú metódu používateľ naposledy klikol a pri prípadnom
            # zobrazení inej vrstvy a následnom zobrazení vrstvy na ktorej bolo kliknuté na niektorú z metód
            # bude táto hodnota už prednastavená.
            method = self.get_checked_method()
            self.__changed_config['config_selected_method'] = method
            if method == 'PCA':

                # Ak je zvolená metóda PCA a je to aj aktuálne používaná metóda pre danú vrstvu, je potrbné zobraziť
                # informácie o výstupe metódy PCA. A skryť iné prípadne zobrazené informácie, konkrétne ak bola
                # predtým zvolená metóda t-SNE, je potrebné skryť kontajner obsahujúci vstupy pre parametre.
                if self.__changed_config['used_method'] == method:
                    self.__pca_info_gbox.show()
                self.__t_sne_parameter_gbox.hide()
            elif method == 't-SNE':

                # Obdobne ako pri metóde PCA je pri zvolení metódy PCA potrebné skryť prípadné informácie o PCA a
                # zobraziť okno s možnosťami zmeny parametrov vstupov pre metódu t-SNE.
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.show()
            else:

                # Ak bola zvolená No method, sú skryté všetky dodatočne grafické prvky, ktoré sa zobrazujú pri metóde
                # PCA alebo t-SNE.
                self.__pca_info_gbox.hide()
                self.__t_sne_parameter_gbox.hide()

    def on_color_label_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox color_labels. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['color_labels'] = self.__color_labels_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_lock_view_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox lock_view. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['locked_view'] = self.__lock_view_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_show_polygon_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox show_polygon. V rámci tejto metódy sa
        aktualizuje konfiguračná premenná, je nastavená premenná pojednávajuca o tom, či majú byť pre danú
        vrstvu prepočítané hodnoty hrán polygonu, ktorý lemuje zobrazené body. Následne je zmenený konfig
        na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config[
                'show_polygon'] = self.__active_layer.calculate_polygon = self.__show_polygon_cb.isChecked()
            if self.__changed_config['show_polygon']:
                self.__active_layer.set_polygon_cords()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def on_3d_graph_check(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá v prípade, ak bolo kliknuté na checkbox 3d_view. V rámci tejto metódy sa aktualizuje
        konfiguračná premenná, zmenený konfig je na aktívnej vrstve použitý a graf je prekreslený, ak aktívna vrstva
        nejaký graf zobrazuje.
        """
        if self.__changed_config:
            self.__changed_config['draw_3d'] = self.__3d_view_cb.isChecked()
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()

    def validate_label_entry(self, identificator, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda sa zavolá po tom, čo bolo stlačené tlačidlo enter pri úprave označení osí zobrazovaných na grafe.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param identificator: identifikátor vstupu, ktorým sa identifikuje, ktorý vstup má byť upravený. ID nadobúda
                              číslo od 0 po 2 a na základe toho je možné určiť os, pre ktorú sa má nazov nastaviť.
                              Osi idú postupne [Os X, Os Y, Os Z]. Id predstavuje index vliste názvov v
                              konfiguračnej premennej.
        :param value: hodnota odoslaná grafickým prvkom, ktorá ma byť nastavená ako označenie pre os.
        """
        self.__labels_entries_list[identificator].set_variable_label(value)
        self.__labels_entries_list[identificator].show_variable_label()
        self.__changed_config['axis_labels'][identificator] = value
        self.__active_layer.use_config()
        self.__active_layer.redraw_graph_if_active()

    def validate_cord_entry(self, identificator, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá overuje, či bol vstup zadaný používateľom valídny vzhľadom na aktuálne použitú metódu na redukciu
        priestoru. Metóda sa zavolá po stlačení enter po napísaní požadovanej súradnice na danej osi.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param identificator: identifikátor stupu, na ktorom bola zadaná hodnota. Je to index do listu držiaceho odkazy
                              na vstupy ako aj index do listu pre nastavenie súradnice pre určitú os.
        :param value:         predstavuje zadanú hodnotu, ktorá sa overuje.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return vracia True alebo False podľa toho, či bol vstup korektný alebo nie.
        """
        # Inicializácia premenných, ktoré budú niesť hraničné hodnoty.
        bottom_border = 0
        top_border = 0
        changed_cords = None

        # V závislosti od aktuálne použitej metódy sa menia prípustné hodnoty a aj tvar zadávaných hodnôt, preto je
        # potrebné to rozlíšiť.
        if self.__cur_used_method == 'No method':

            # Ak sa nepoužíva žiadna metóda na redukciu priestoru, je potrebné zistiť, či má vrstva feature mapy. Ak áno
            # vstup bude načítaný v odlišnom tvare ako v ostatných prípadoch.
            if self.__changed_config['has_feature_maps']:

                # V prípade ak má vrstva feature mapy je hodnota testovaná odlišne. Najskôr sa získa referencia na list
                # obsahujúci zobrazované súradnice. Následne sa do premennej priradí aj výstupný tvar vrstvy, ktorý
                # sa použije pri validácií. Na základe premennej obsahujúcej výstupný tvar vrstvy sa nastaví sa
                # do premennej borders načítajú horné hraničné hodnoty. Na prvý index sa načíta hodonota riad-
                # ky, na druhú pozíciu sa načíta horná hranica pre stĺpce. Ďalej sú definované premenné, kto-
                # ré budú obsahovať prípadný chbový text alebo novú hodnotu.
                changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                output_shape = self.__changed_config['output_shape']
                borders = [output_shape[-3] - 1, output_shape[-2] - 1]
                correct_input = []
                output_text = ['', '']

                # Vstup od používateľa je následne rozdelený na základe delimetra čiarky. Je potrebné, aby používateľ
                # dodržal predpísaný tvar vstupu v tvare (riadok, stĺpec). Nadbytočné medzeri by nemali byť problém
                # no dôležité je použitie čiarky. Podľa toho, koľko prvkov bude obsahovať list po rozdelení sa
                # rozhodne, ako sa bude ďalej pokračovať.
                entry_input = value.split(',')
                if len(entry_input) == 2:

                    # Ak bol počet prvkov po rozdelení 2, pokračuje sa testovaním, či sú obe hodnoty číselné a či sa na-
                    # chádzajú v prípustnom intervale. Toto testovanie prebieha v cykle a akonáhle nastane výnimka ale-
                    # bo zadaná hodnota neleží v prípustnom intervale, je do listu pre výpis vložená chybová hláška.
                    # Ak však výnimka nenastane a hodnota leží v požadovanom intervale, je táto hodnota vložená do
                    # listu pre výpis.
                    for i, number in enumerate(entry_input):
                        try:
                            number_int = int(number)
                            if 0 <= number_int <= borders[i]:
                                correct_input.append(int(number))
                                output_text[i] = int(number)
                            else:
                                output_text[i] = 'err'
                        except ValueError:
                            output_text[i] = 'err'

                    # Na základe hodnôt v liste, ktorý je určený pre výstup je poskladaná výsledná správa.
                    output_msg = f'{output_text[0]}, {output_text[1]}'
                    # Ak sa počet správnych vstupov rovná dvom, všetko prebehlo správne a je možné nastaviť do referen-
                    # cie zobrazovaných súradníc načítané hodnoty. Hodnoty sú zapísané do listu hodnôt na os, ktorá je
                    # určená identifikátorom vstupu.
                    if len(correct_input) == 2:
                        changed_cords[0][identificator] = correct_input[0]
                        changed_cords[1][identificator] = correct_input[1]
                        self.__cords_entries_list[identificator].set_entry_text(output_msg)
                        return_val = True
                    else:
                        return_val = False
                else:

                    # V prípade ak sa počet zadnaých vstupov nerovná dvom je nastavená nasledujúca výstupná správa a
                    # hodnota návratovej premennej na hodnotu False.
                    output_msg = 'err'
                    return_val = False

                # Po vyhodnotení toho, či bol vstup valídny alebo nie sú nastavené zobrazované hodnoty na grafických
                # prvkoch.
                if return_val:
                    # Ak bol vstup valídny je zobrazená nova hodnota súradníc, následne je signalizovaná zmena
                    # zobrazovaných súradníc a je použitá zmenená konfiguračná premenná na aktívnej vrstve.
                    # Taktiež je prekreslený graf.
                    self.__cords_entries_list[identificator].set_variable_label(output_msg)
                    self.__cords_entries_list[identificator].show_variable_label()
                    self.__changed_config['cords_changed'] = True
                    self.__active_layer.use_config()
                    self.__active_layer.redraw_graph_if_active()
                else:
                    # Ak vstup nebol správny, je ponechaný zobrazený vstup a je nastavená výstupná správa, ktorá môže
                    # ukazovať, ktoré zo zadaných čísel bolo nesprávne. Ak sa objaví iba jedna chybova hĺaška znamená
                    # to s najväčšou pravdepodobnosťou nesprávny formát vstupu.
                    self.__cords_entries_list[identificator].set_entry_text(output_msg)
                # Následne je vrátená hodnota o úspešnosti vstupu.
                return return_val
            else:

                # Ak nebola zvolená žiadna metóda redukcie priestoru a vrstva nemá feature mapy, je nastavená spodná a
                # horná hranica pre validáciu prípustných hodnôt. Ak je to možné je vstupná hodnota od používateľa
                # prevedená na int, ak pri pretypovaní dôjde k chybe je na ponechaný zobrazený element na zadanie
                # vstupu a je doň zapísaná hodnota značiaca chybu.
                bottom_border = 0
                top_border = self.__changed_config['output_dimension']
                changed_cords = self.__changed_config['no_method_config']['displayed_cords']
                try:
                    new_value = int(value)
                except ValueError:
                    self.__cords_entries_list[identificator].set_entry_text('err')
                    return False
        elif self.__cur_used_method == 'PCA':
            # Ak bola je používaná metóda PCA, budú hraničné hodnoty a taktiež aj hodnota uložené do premenných. Pri
            # metóde PCA sú hranice aj hodnota trochu odlišná, lebo komponenty sa zadávajú od čísla 1. Od
            # používateľa príjme hodnotu väčšiu alebo rovnú ako 1, no na pozadí je hodnota odčítaná aby
            # zadaná hodnota mohla predstavovať index do poľa zobrazovaných súradníc. Následne je
            # pretypovaná na int. Ak sa pri pretypovaní neobjaví výnimka, pokraćuje sa ďalej, ak
            # áno je nastavená hodnota err ako text do zobrazovaného vstupu.
            bottom_border = 1
            top_border = min(self.__changed_config['output_dimension'],
                             self.__changed_config['number_of_samples']) + 1
            changed_cords = self.__changed_config['PCA_config']['displayed_cords']
            try:
                new_value = int(value) - 1
            except ValueError:
                self.__cords_entries_list[identificator].set_entry_text('err')
                return False
        elif self.__cur_used_method == 't-SNE':
            # Ak sa použije metóda t-SNE, sú podľa konfigurácie nastavené hraničné hodnoty. Zadaná hodnota je následne
            # pretypovaná na typ int. Ak sa pri pretypovaní objaví chyba, je do vstupu písaná hodnota err.
            bottom_border = 0
            top_border = self.__changed_config['t_SNE_config']['used_config']['n_components']
            changed_cords = self.__changed_config['t_SNE_config']['displayed_cords']
            try:
                new_value = int(value)
            except ValueError:
                self.__cords_entries_list[identificator].set_entry_text('err')
                return False

        # Na základe nastavených hodnôt a hraníc sa vyhodnotí, či je vstup valídny. Ak áno je nová hodnota priradená
        # do konfiguračnej premennej a sú nastavené aj novo zobrazované hodnoty. Ak bol vstup valídny je signali-
        # zovaná zmena vstupu a následne je konfiguračná premenná pre aktívnu vrstvu použítá. Na záver je
        # prekreslený graf.
        try:
            if not (bottom_border <= int(value) < top_border):
                self.__cords_entries_list[identificator].set_entry_text('err')
                return False

            self.__cords_entries_list[identificator].set_variable_label(value)
            self.__cords_entries_list[identificator].show_variable_label()
            changed_cords[identificator] = int(new_value)
            self.__changed_config['cords_changed'] = True
            self.__active_layer.use_config()
            self.__active_layer.redraw_graph_if_active()
            return True
        except ValueError:
            self.__cords_entries_list[identificator].set_entry_text('err')
            return False

    def validate_t_sne_entry(self, identificator, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Validuje vstup pre parametre metódy t-SNE.

        Parmetre
        ----------------------------------------------------------------------------------------------------------------
        :param identificator: identifikátor vstupu.
        :param value:         hodnota zadaná na vstupe

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: navracia True alebo False na základe toho, či bol zadaný vstup valídny.
        """
        # Všetko prebieha v try catch blocku. Identifikator vstupu je aj kľúčom k odkazu na grafický prvok pre vstup
        # od užívateľa ako aj kľúč v rámci konfiguračného súboru, kde sú uložené prípustné hranice pre parameter
        # a taktiež a typová hodnota parametra.
        try:
            if self.__changed_config is not None:

                # Na základe id je do premennej test_tuple priradený tuple obsahujúci na prvej pozícií spodnú hranicu
                # prípustného intervalu, na druhej pozicií prípustný typ zadanej hodnoty a na tretej pozicií hornú
                # hranicu intervalu. Zadaná hodnota je najskôr pretypovaná na prípustný typ. Ak nedôjde pri
                # pretypovaní k chybe, je hodnota porovnaná s hraničnými hodnotami. Ak zadaná hodnota leží
                # v prípustnom intervale, je hodnota priradená do konfiguračnej premennej. AK nie, je
                # na vstupe vypísaná chybová hláška. Ak je nová hodnota odlišná od posledne použitej
                # hodnoty, označí sa grafický prvok ako zmenený, to znamená, že sa zmení jeho farba
                # na červenú a k jeho názvu je pridaná hviezidčka.
                test_tuple = self.__changed_config['t_SNE_config']['parameter_borders'][identificator]
                if not (test_tuple[0] <= test_tuple[1](value) <= test_tuple[2]):
                    self.__tSNE_parameters_dict[identificator].set_entry_text('err')
                    return False
                parameter_label = self.__tSNE_parameters_dict[identificator]
                parameter_label.set_variable_label(value)
                parameter_label.show_variable_label()
                self.__changed_config['t_SNE_config']['options_config'][identificator] = test_tuple[1](value)
                if self.__changed_config['t_SNE_config']['options_config'][identificator] == \
                        self.__changed_config['t_SNE_config']['used_config'][identificator]:
                    parameter_label.set_mark_changed(False)
                else:
                    parameter_label.set_mark_changed(True)
                return True
            else:
                return False
        except ValueError:
            self.__tSNE_parameters_dict[identificator].set_entry_text('err')
            return False

    def get_checked_method(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vracia na základe zakliknutého radio buttonu názov metódy.

        Návratová hodnota
        ----------------------------------------------------------------------------------------------------------------
        :return: string nesúci názov použitej metódy
        """
        if self.__no_method_rb.isChecked():
            return 'No method'
        elif self.__PCA_rb.isChecked():
            return 'PCA'
        elif self.__t_SNE_rb.isChecked():
            return 't-SNE'
        return None

    @property
    def active_layer(self):
        return self.__active_layer
