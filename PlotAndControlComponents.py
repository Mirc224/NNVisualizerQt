import gc
import math
import threading

import matplotlib
import mpl_toolkits.mplot3d as plt3d
import numpy as np
from matplotlib import backend_bases
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from QtGraphicalComponents import *

matplotlib.use('Qt5Agg')

# Dfeinovanie zobrazovanýh itemov v toolbare grafu.
backend_bases.NavigationToolbar2.toolitems = (
    ('Home', 'Reset original view', 'home', 'home'),
    ('Back', 'Back to  previous view', 'back', 'back'),
    ('Forward', 'Forward to next view', 'forward', 'forward'),
    ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
    ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
)


class PlottingFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda, ktorej hlavnou úlohou je vykresľovať poskytnuté dáta. Je možné vykresľovať dáta v 2D alebo 3D. Umožňuje
        zobaziť body ako aj hrany priestorovej mriežky, na základe začiatočných a koncových bodov.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.__cords:        atribút obsahuje súradnice bodov, ktoré majú byť zobrazené. V závislosti od poskytnu-
                                  tých dát obsahuje jeden, dva alebo tri vnorené polia, obsahujúce súradnice bodov na
                                  jednotlivých vrstvách.
        :var self.__line_cords_t: ak je možné zobraziť priestorovú mriežku, obsahuje tento atribút dvojice transformova-
                                  ných začiatočných a koncových bodov.
        :var self.__n_of_output:  atribút nesie informáciu o tom, koľko môže byť na výstupe dimenzií.
        :var self.__n_of_displ:   udáva počet dimenzií, ktoré sú zobazované.
        :var self.__graph_title:  obsahuje názov grafu, ktoré sa bude zobrazovať nad figúrou.
        :var self.__axis_labels:  obsahuje označenia jednotlivých osí grafu.
        :var self.__master:       odkaz na nadradenú triedu, ktorej budú odosielané signály o požadovaných zmenách.
        :var self.__pts_config:   odkaz na konfiguráciu načítnaých bodov.
        :var self.__points_color: zoznam farieb pre jednotlivé body.
        :var self.__figure:       odkaz na inštanciu matplotlib figúru.
        :var self.__canvas:       plátno, na ktorom je graf vykresľovaný.
        :var self.__axis:         odkaz na triedu Axes, ktorá v sebe drží informácie a nastavenia o zobrazovaných
                                  osiach.
        :var self.__toolbar:      matplotlib toolbar, umožňujúci ovládanie pri 2D grafe.
        :var self.__color_labels: informuje o tom, či majú byť v grafe body ofarbené v závislosti od triedy, ak je to
                                  možné.
        :var self.__draw_polygon: atribút poskytuje informáciu o tom, či sa má vykresliť priestorová mriežka, ak je to
                                  možné.
        :var self.__locked_view:  nesie informáciu o tom, či má byť uzamknutý pohľad grafu, to znamená či sa má pri pre-
                                  kresľovaní prispôsobovať zobrazovaným dátam, alebo si zachovať nastavenia pohľadu.
        :var self.__draw_3D:      predstavuje informáciu o tom, či sa má vykresľovať 2D alebo 3D graf.
        :var self.__cond_var:     podmienková premenná, ktorá zabezpečuje synchronizáciu. Zaručuje, aby pri vykresľovaní
                                  grafu boli dostupné všetky požadované atribúty.
        """
        super(PlottingFrame, self).__init__(*args, **kwargs)

        # Definovanie základnýh atribútov.
        self.__cords = None
        self.__master = None
        self.__n_of_displ = -1
        self.__n_of_output = None
        self.__line_cords_t = None

        # Definícia atribútrov ohľadom názvu grafu a označení osí grafu.
        self.__graph_title = 'Graf'
        self.__axis_labels = ['Label X', 'Label Y', 'Label Z']

        # Definícia atribútov ohľadom vzhľadu a zobrazenia načítaných bodov.
        self.__pts_config = None
        self.__points_color = None

        # Definícia matplotlib figúry, definovanie plátna pre vykresľovanie a nastavenie figúry. Vytvorenie inštancie
        # toolbaru pre manipuláciu s 2D grafmi.
        self.__figure = Figure(figsize=(4, 4), dpi=100)
        self.__canvas = FigureCanvasQTAgg(self.__figure)
        self.__axis = self.__figure.add_subplot(111, projection='3d')
        self.__toolbar = NavigationToolbar(self.__canvas, self)

        # Definovanie atribútov informujúcich o možnostiach zobrazenia v grafe.
        self.__draw_3D = False
        self.__locked_view = False
        self.__color_labels = False
        self.__draw_polygon = False

        # Vytvorenie podmienkovej premennej a zavolanie metódy na grafickú inicializáciu.
        self.__cond_var = threading.Condition()
        self.initialize_ui()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Zabezpečuje inicializáciu a rozmiestnenie grafických prvkov.
        """

        # Je vyytvorené hlavné rozmiestnenie, ktoré je nastavné triede. Do tohto rozmiestnenia je pridané najskôr plát-
        # no, ktoré zabezpečuje vykresľovanie grafu a následne toolbar pre manipuláciu s 2D grafom. Na plátno je
        # pripijený signál, ktorý sa vyvolá pri udalosti kliknutia.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__canvas)
        layout.addWidget(self.__toolbar)
        self.__toolbar.setMinimumHeight(35)
        self.__canvas.mpl_connect('button_press_event', self.on_mouse_double_click)

    def initialize(self, controller, number_of_outputs, displayed_cords, points_config: dict,
                   layer_name: str):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia grafu na zákalde poskytnutých parametrov. Je inicializovaný zoznam farieb použitých pri vykresľo-
        vaní bodov, nadradený prvok prijmajúci signály o zmenách v grafe, načítané body na zobrazenie, odkaz na konfi-
        guráciu načítaných bodov, informácia o výstupnej dimenzií neurónovej siete, ktorej výstup má graf zobrazovať.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param controller:        odkaz na inštanciu triedy, ktorá bude prijmať signaly vziknuté v grafe.
        :param number_of_outputs: udáva počet výstupov, ktoré sú na neurónovej vrstve.
        :param displayed_cords:   pole súradníc, ktoré majú byť vykreslené.
        :param points_config:     odkaz na konfiguráciu zobrazovaných bodov.
        :param layer_name:        nadpis vykresľovaného grafu.
        :return:
        """
        self.__points_color = []
        self.__master = controller
        self.__cords = displayed_cords
        self.__graph_title = layer_name
        self.__pts_config = points_config
        self.__n_of_output = number_of_outputs
        self.set_graph_dimension(self.__n_of_output)

    def set_graph_dimension(self, dimension: int):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastaví týp grafu, ktorý sa má zobrazovať. Môže ísť o 2D alebo 3D graf.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param dimension: obsahuje počet dimenzií.
        """
        self.__figure.clf()
        # Podľa poskytnutého údaju o dimenziách treba určiť, aký typ grafu sa má vykresliť.
        if dimension >= 3:
            # Ak je hodnota parametra väčšia alebo rovná trom, nastaví sa hodnota parametra draw_3D na hodnotu True, čo
            # znamená, že má byť vykreslený 3D graf. Atribútu nesúcemu informáciu o počte zobrazovaných dimenzií je
            # priradené číslo 3. Ďalej je získaný odkaz na triedu Axes inicializovanú na zobrazovanie 3D grafov.
            self.__draw_3D = True
            self.__n_of_displ = 3
            self.__axis = self.__figure.add_subplot(111, projection='3d')

            # Pri vykresľovaní 3D grafu sú skryté všetky tlačidla, okrem grafických labels, ktoré informujú o súradni-
            # ciach, nad ktorými je kurzor. Tlačidlá toolbaru pri vykresľovaní 3D grafu nefungujú ako pri 2D grafe,
            # preto je zbytočné ich zobrazovať.
            for child in self.__toolbar.children():
                if not isinstance(child, QLayout):
                    if not isinstance(child, QLabel) and not isinstance(child, QWidgetAction):
                        child.setVisible(False)
        else:
            # Ak je hodnota parametra menšia ako 3, bude vykreslený 2D graf. Je nastavená hodnota atribútu hovoriaca o
            # počte zobrazovaných dimenzií. Pomocou figúry je získaná inštancia triedy Axes nastavená na zobrazovanie
            # 2D grafov.
            self.__draw_3D = False
            self.__n_of_displ = 2
            self.__axis = self.__figure.add_subplot(111)

            # Ak boli náhodou tlačidlá toolbaru skryté, budú v rámci cyklu zobrazené. Tlačidlá toolbaru umožňujú ovláda-
            # nie pohľadu 2D grafu.
            for child in self.__toolbar.children():
                if not isinstance(child, QLayout):
                    if not isinstance(child, QLabel) and not isinstance(child, QWidgetAction):
                        child.setVisible(True)

    def redraw_graph(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V rámci tejto metódy je vykreslený prekreslený graf a to v súlade s požadovanými nastaveniami zobrazenia.
        """

        # Pomocou podmienkovej premennej je zaistená synchronizácia, aby náhodou nenastala situácia, že bude graf zmaza-
        # ný počas toho, ako bude prebiehať jeho prekresľovanie.
        self.__cond_var.acquire()
        if self.__canvas is not None:

            if self.__locked_view:
                # Ak je zvolená možnosť na uzamknutie pohľadu, sú do pomocných premenných uložené nastavenia aktuálne
                # zobrazovaného pohľadu pre osu X a osu Y.
                tmpX = self.__axis.get_xlim()
                tmpY = self.__axis.get_ylim()
                if self.__n_of_displ == 3:
                    # Ak sú na výstupe zobrazované 3 dimenzie, je do pomocnej premennej uložné aj nastavenie zobrazenia
                    # pre osu Z.
                    tmpZ = self.__axis.get_zlim()

            # Je vyčistená inštancia triedy Axes a vykreslená mriežka na stenách grafu.
            self.__axis.clear()
            self.__axis.grid()

            # Nastavenie zobrazovaného nadpisu grafu a označenia osi X.
            self.__axis.set_title(self.__graph_title)
            self.__axis.set_xlabel(self.__axis_labels[0])

            # Podľa počtu vrstvy výstupov, ktoré má graf zobrazovať sú prípadne pomenované aj ďalšie osi.
            if self.__n_of_output > 1:
                # Ak je počet možných výstupov vrstvy viac ako 1, je nastavené aj označenie osi Y.
                self.__axis.set_ylabel(self.__axis_labels[1])
                if self.__draw_3D:
                    # Ak je požadované vykreslenie 3D grafu, je použité označenie aj pre os Z.
                    self.__axis.set_zlabel(self.__axis_labels[2])

            # Je potrebné overiť, či sú dostupné body na zobrazenie.
            if self.__cords is not None:
                # Podľa rozmeru premennej obsahujúcej súradnice bodov je zistený počet poskytnutých súradníc.
                number_of_cords = len(self.__cords)

                # Podľa atribútu draw_polygon sa rozhodne, či má byť vykreslená priestorová mriežka.
                if self.__draw_polygon:
                    # V prípade, že má byť vykreslená priestorová mriežka je podľa počtu vykresľovaných dimenzií zvolený
                    # ďalší postup.
                    if self.__n_of_displ == 3:
                        # Ak je počet zobrazovaných dimenzií rovný 3, sú v rámci cyklu vytvárané usporiadané dvojice
                        # začiatočných a koncových súradníc pre jednotlivé osi, na zákalde čoho sú čiary vykreslené.
                        for edge in self.__line_cords_t:
                            xs = edge[0][0], edge[1][0]
                            ys = edge[0][1], edge[1][1]
                            zs = edge[0][2], edge[1][2]
                            line = plt3d.art3d.Line3D(xs, ys, zs, color='black', linewidth=1, alpha=0.3)
                            self.__axis.add_line(line)
                    if self.__n_of_displ == 2:
                        # Ak je počet zobrazovaných súradíc dva sú vytvorené usporiadané dvojice začiatočných a konco-
                        # vých bodov pre jednotlivé osi.
                        for edge in self.__line_cords_t:
                            xs = edge[0][0], edge[1][0]
                            # V závislosti od počtu poskytnutých súradníc sú vytvorené usporiadané dvojice začiatočných
                            # a koncových súradníc pre os Y.
                            if number_of_cords == 1:
                                ys = 0, 0
                            else:
                                ys = edge[0][1], edge[1][1]
                            # Postupne sú čiary vykresľované.
                            self.__axis.plot(xs, ys, linestyle='-', color='black', linewidth=1, alpha=0.5)

                # Ak boli poskytnuté nejaké súradnice na zobrazenie, je zrejmé, že premenná obsahuje aspoň jedno pole
                # súradníc. Na základe tohto poľa sú pre ďalšie osi vytvorené rovnako dlhé polia naplnené nulami.
                x_axe_cords = self.__cords[0]
                y_axe_cords = np.zeros_like(self.__cords[0])
                if self.__draw_3D:
                    z_axe_cords = np.zeros_like(self.__cords[0])

                # Je nastavená farba, ktorá má byť pre jednotlivé body použitá.
                self.set_point_color()

                # Podľa počtu poskytnutých súradníc sú do pomocných premenných priradené príslušné hodnoty.
                if number_of_cords >= 2:
                    y_axe_cords = self.__cords[1]
                    if number_of_cords > 2:
                        z_axe_cords = self.__cords[2]

                # Nasleduje vykreslenie bodov grafu. To sa líši od toho, či ide o 2D alebo 3D graf.
                if self.__draw_3D:
                    self.__axis.scatter(x_axe_cords, y_axe_cords, z_axe_cords, c=self.__points_color)
                    # Ak sú aktivované niektoré z bodov a bola im priradená trieda, je táto trieda vykreslená pri zodpo-
                    # vedajúcom bode.
                    for point in self.__pts_config['active_labels']:
                        self.__axis.text(x_axe_cords[point],
                                         y_axe_cords[point],
                                         z_axe_cords[point],
                                         self.__pts_config['label'][point])
                else:
                    # Vykreslenie bodov v 2D grafe.
                    self.__axis.scatter(x_axe_cords, y_axe_cords, c=self.__points_color)
                    # Vedľa zvolených aktívnych bodov je vypísaný názov ich triedy.
                    for point in self.__pts_config['active_labels']:
                        if point < len(self.__pts_config['label']):
                            self.__axis.annotate(self.__pts_config['label'][point],
                                                 (x_axe_cords[point], y_axe_cords[point]))

            if self.__locked_view:
                # Ak je zvolená možnosť uzamknutia pohľadu, je pomocou vyšie definovaných premenných použité predchádza-
                # úce nastavenia.
                self.__axis.set_xlim(tmpX)
                self.__axis.set_ylim(tmpY)
                if self.__n_of_displ == 3:
                    self.__axis.set_zlim(tmpZ)
            # Na záver je prekreslené plátno obsahujúce zobrazenie upravenej figúry.
            self.__canvas.draw()

        # Podmienková premenná notifikuje o tom, že bol uvoľnený zámok.
        self.__cond_var.notify()
        self.__cond_var.release()

    def set_color_label(self, new_value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Nastavenie píznaku o tom, či majú byť body priradené podľa príslušnosti k triede.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param new_value: parameter obsahuje bool hodnotu, hovoriacu o tom, ako má byť príznak nastavený.
        """
        if new_value:
            # Ak má byť testovaný príznak nastavený na True, je potrebné existovať, či vôbec existuje v rámci konfigu-
            # rácie bodov informácia o farbá podľa príslušnosti k triede.
            if self.__pts_config['label_color'] is not None:
                self.__color_labels = True
            else:
                self.__color_labels = False
        else:
            self.__color_labels = False

    def set_point_color(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        V metóde je do listu farieb zobrazovaných bodov, každému bodu nastavená farba, ktorou má byť pri vykreslení bod
        ofarbený. Farba závisí od použitých nastavení. Bod je ofarbený buiď to základnou farbou použitou defaultne pre
        všetky body alebo farbou podľa príslušnosti k triede. Ak bol bod dvojkliknutím aktivovaný je mu nastavená
        farba aktívneho bodu.
        """
        # Testuje sa, či je zvolená možnosť ofarbenia bodov, podľa príslušnosti k triede.
        if self.__color_labels:
            # Ak je táto možnosť zvolená, testuje sa, či existuje v rámci konfiguračnej premennej pole farieb bodov
            # podľa príslušnosti k triede.
            if self.__pts_config['label_color'] is not None:
                # Ak toto pole existuje, je do atribútu points_color priradená kópia tohto zoznamu.
                self.__points_color = self.__pts_config['label_color'].copy()
            else:
                # Ak nie, je do tohto atribútu priradená kópia poľa základných farieb.
                self.__points_color = self.__pts_config['default_color'].copy()
        else:
            # Ak táto možnosť nie je zvolená, je do poľa použitých farieb bodov vložená kópia poľa základných farieb.
            self.__points_color = self.__pts_config['default_color'].copy()

        # Niektoré z bodov môžu byť aj aktivované dvojklikom, preto je tento zoznam bodov uložený v konfiguračnej preme-
        # nnej prejdený v rámci cyklu a aktívnym bodom je priradená príslušná farba.
        for point, color in self.__pts_config['different_points_color']:
            self.__points_color[point] = color

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda uvoľní prostriedky používané touto triedou. Je potrebné zaistiť synchronizáciu aby neboli prvky zmazané
        v čase, keď bude v inom vlákne graf vykresľovaný. Na to poslúži podmienková premenná.
        """
        self.__cond_var.acquire()
        self.__figure.clear()
        self.__figure.clf()
        self.__axis.cla()
        self.__canvas.close()
        self.__canvas.deleteLater()
        self.__toolbar.close()
        self.__toolbar.deleteLater()
        self.deleteLater()
        self.__canvas = None
        self.__toolbar = None
        self.__figure = None
        self.__axis = None
        self.__cond_var.notify()
        self.__cond_var.release()
        gc.collect()

    def on_mouse_double_click(self, event):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda, ktorá sa zavolá po kliknutí na graf. V tejto metóde sa zistí či bolo kliknuté na niektorý z vykresľova-
        ných bodov a v prípade, ak áno je tento bod vyznačený a taktiež je prípadne zobrazená trieda, ku ktorej prislú
        cha. Pomocou konfiguračnej premennej je zmena dostupná všetkým zobrazeným grafom.

        Popis
        ----------------------------------------------------------------------------------------------------------------
        :param event: obsahuje informácie o udalosti, ktorá sa vyskytla. Pomocou nej je možné zistiť aké tlačidlo bolo
                      stlačené, prípadne súradnice myši, na ktorých došlo ku kliknutiu.
        """
        # Metóda sa zavola aj pri jednorazovom kliknutí na graf. To by bolo nepraktické, pretože klikaním je ovládaná
        # rotácia pri 3D grafe, preto je potrebné zistiť, či došlo ku dvojkliku.
        if event.dblclick and event.button == 1:
            # Ak došlo k dvojkliku na lavom tlačidle, sú vyčistené polia v konfiguračnej premennej bodov.
            self.__pts_config['different_points_color'].clear()
            self.__pts_config['active_labels'].clear()

            # Do pomocnej premennej je definovaná hodnota v pixeloch v rámci ktorej bude považované, že došlo ku klik-
            # nutiu na určitý bod, pretože trafiť presne stred bodu je takmer nemožné. Do pomocnej premennej closest_-
            # point je priradená hodnota neplatného index, aby bolo možné zistiť, či naozaj došlo ku kliknutiu na
            # valídny bod.
            nearest_x = 3
            nearest_y = 3
            closest_point = -1
            # Ak nie sú nastavené žiadne súradnice bodov, nemá význam zisťovať na aký bod bolo kliknuré a metóda končí.
            if self.__cords is not None:
                # Ak však sú priradené súradnice bodov, je zistené koľko súradníc je dostupných. Ak sú nejaké body pri-
                # radené, je zrejmé, že existuje súradnica aspoň pre os X. Nasledne je potrebné zistiť, či boli posky-
                # tnuté aj iné súradnice.
                number_of_points = len(self.__cords[0])
                X_point_cords = self.__cords[0]
                if len(self.__cords) > 1:
                    # Ak bol počet poskytnutých súradníc viac ako jedna, vyplýva z toho, že bola poskytnutá aj Y súrad-
                    # nica zobrazovaných bodov.
                    Y_point_cords = self.__cords[1]
                else:
                    # Ak nebola poskytnutá druhá súradnica, je do pomocnej premennej priradené pole o rovnakej veľkosti
                    # ako bolo pole obahujúce prvú súradnicu a je naplnené nulami.
                    Y_point_cords = np.zeros_like(self.__cords[0])

                if self.__draw_3D:
                    # Ak je nastavená možnosť vykresľovania 3D je potrebné získať aj hodnoty pre 3tiu súradnice.
                    if len(self.__cords) == 3:
                        # Ak boli poskytnuté 3 súradnice, je do pomocnej premennej priradené pole týchto súradníc.
                        Z_point_cords = self.__cords[2]
                    else:
                        # Ak nebola poskytnutá 3tia súradnica je vytvorené pole o rovnakej veľkosti ako pole obsahujúce
                        # prvú súradnicu, ktoré je naplnené nulami.
                        Z_point_cords = np.zeros_like(self.__cords[0])

                # V nasledujúcom cykle je pre všetky body testované, či došlo k ich aktivácií.
                for point in range(number_of_points):
                    # Výpočet súradníc projekcie na obrazovku závisí od vykresľovaného grafu.
                    if self.__draw_3D:
                        # V prípade 3D grafu, sú najskôr všety súradnice bodov transfomované do 2D pomocou transformač-
                        # nej matice. Z 2D súradníc dokážeme potom získať súradnice týchto bodov na obrazovke.
                        x_2d, y_2d, _ = proj3d.proj_transform(X_point_cords[point], Y_point_cords[point],
                                                              Z_point_cords[point], self.__axis.get_proj())
                        pts_display = self.__axis.transData.transform((x_2d, y_2d))
                    else:
                        # Ak je vykresľovaný 2D graf, sú súradnice grafu prevedené na obrazovkové súradnice a tieto sú
                        # uložené v pomocnej premennej.
                        pts_display = self.__axis.transData.transform((X_point_cords[point], Y_point_cords[point]))

                    # Následne sa porovnáva či došlo ku kliknutiu v oblasti, ktorú možno považovať za oblasť bodu na zá-
                    # klade získaných obrazovkových súradníc.
                    if math.fabs(event.x - pts_display[0]) < 3 and math.fabs(event.y - pts_display[1]) < 3:
                        # Ak došlo ku kliknutiu v oblasi, ktorú môžeme považovať za oblasť bodu, nasleduje zisťovanie či
                        # je tento bod bližšie ako doposiaľ najbližší bod tomuto kliknutiu.
                        if nearest_x > math.fabs(event.x - pts_display[0]) and nearest_y > math.fabs(
                                event.y - pts_display[1]):
                            # Ak áno, sú nastavené hodnoty najmenšej vzdialenosti od kliknutia a aj index bodu, ktorému
                            # tieto vzdialenosti prislúchajú.
                            nearest_x = math.fabs(event.x - pts_display[0])
                            nearest_y = math.fabs(event.y - pts_display[1])
                            closest_point = point

                # Testuje sa, či bol nájdený valídny index nejkého bodu.
                if closest_point != -1:
                    # Ak pomocnná premenná obsahuje číslo rozličné od mínus jedna, znamená to, že obsahuje index bodu
                    # na ktorý bolo kliknuté. Preto je do konfiguračnej premennej voložená dvojica obsahujúca index
                    # bodu a aktivačnú farbu.
                    self.__pts_config['different_points_color'].append((closest_point, '#F25D27'))

                    # Ďalej sa testuje, či v konfiguračnej premennej existuje pole obsahujúce triedy prislúchajúce je-
                    # dnotlivým bodom podľa ich indexu. Zároveň je potrebné zistiť, či je počet tried rovný počtu
                    # načítaných bodov, aby nedošlo k chybe.
                    if self.__pts_config['label'] is not None and \
                            len(self.__pts_config['label']) == self.__pts_config['number_of_samples']:

                        self.__pts_config['active_labels'].append(closest_point)
                # Na záver je zaslaný signál na prekreslenie všetkých zobrazených grafov. To zaručí, že bude daný bod
                # vyznačený na všetkých grafoch.
                self.__master.require_graphs_redraw()

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
        return self.__axis_labels

    @graph_labels.setter
    def graph_labels(self, new_labels_list):
        self.__axis_labels = new_labels_list

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
        self.__cond_var.acquire()
        self.__draw_polygon = value
        self.__cond_var.release()

    @property
    def is_3d_graph(self):
        return self.__draw_3D

    @is_3d_graph.setter
    def is_3d_graph(self, value):
        if self.__draw_3D != value:
            self.__cond_var.acquire()
            self.__draw_3D = value
            self.__cond_var.release()
            if self.__draw_3D:
                self.set_graph_dimension(3)
            else:
                self.set_graph_dimension(2)

    @property
    def points_cords(self):
        return self.__cords

    @points_cords.setter
    def points_cords(self, new_cords):
        self.__cond_var.acquire()
        self.__cords = new_cords
        self.__cond_var.release()

    @property
    def line_tuples(self):
        return self.__line_cords_t

    @line_tuples.setter
    def line_tuples(self, new_tuples):
        self.__cond_var.acquire()
        self.__line_cords_t = new_tuples
        self.__cond_var.release()

    @property
    def point_color(self):
        return self.__points_color

    @point_color.setter
    def point_color(self, new_value):
        self.__cond_var.acquire()
        self.__points_color = new_value
        self.__cond_var.release()


class LayerWeightControllerFrame(QWidget):
    def __init__(self, *args, **kwargs):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Ide o triedu, od ktorej budú dediť triedy NoFMWeightControllerFrame a FMWeightControllerFrame. Obsahuje grafické
        prvky, pomocou ktorých sú zobrazované a menené váhy príslušnej neurónovej siete.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self._master:          odkaz na nadradenú triedu, ktorej budú zasielané informácie o zmenách.
        :var self._bias_ref:        obsahuje referenciu pole bias hodnôt príslušnej neurónovej vrstvy.
        :var self._weights_ref:     referncia na pole váh príslušnej neurónovej vrstvy.
        :var self._slider_dict:     slovník konfigurácií pre vytvorenie slidrov. Kľúčom do tohto slovníka je unikátne
                                    meno slidera.
        :var self._pos_n_of_sliders:celkový počet sliderov, ktoré je možné zobraziť.
        :var self._act_slider_dict: slovník obsahujúci referencie na vytvorené a aktívne slidre. Kľúčom je unikátne meno
                                    slidera.
        :var self._scroll_win:      odkaz na grafický komponen obsahujúci aktívne slidre a AddRemoveCombobox pre ich
                                    pridávanie.
        :var self._scroll_area_c:   widget, ktorý v sebe drží obsah scrollovacieho okna, tak aby sa pri dosiahnutí maxi-
                                    málnej možnej kapacity objavil scrollbar a bolo možné v tomto okne scrollovať.
        :var self._scroll_area_l:   rozmiestnenie v rámci obsahu scrollovacieho okna.
        :var self._add_slider_rc:   inštancia triedy AddRemoveCombobox obsahujúca názvy (unikátne kľúče), sliderov, kto-
                                    ré môžu byť zobrazené.
        """
        super(QWidget, self).__init__(*args, **kwargs)
        # Definovanie základných atribútov.
        self._master = None
        self._bias_ref = None
        self._weights_ref = None

        # Definovanie atribútov týkajúcich sa informácií o slideroch a ich zobrazení.
        self._slider_dict = {}
        self._pos_n_of_sliders = 0
        self._act_slider_dict = {}

        # Definovanie grafických prvkov.
        self._scroll_win = QScrollArea(self)
        self._scroll_area_c = QWidget(self._scroll_win)
        self._scroll_area_l = QVBoxLayout(self._scroll_area_c)
        self._add_slider_rc = RemovingCombobox(self._scroll_area_c)

        # Následne prebehne grafická inicializácia.
        self.initialize_ui()

    def initialize_ui(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Inicializácia grafických prvkov a ich rozmiestnenia.
        """
        # Je vytvorené hlavné rozmiestnenie, ktoré je nastavené nadradenému QWidgetu.
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scrollovaciemu oknu je nastavená schopnosť zmeny veľkosti a element je pridany do hlavného rozmiestnenia.
        self._scroll_win.setWidgetResizable(True)
        layout.addWidget(self._scroll_win)

        # Sú definované vlastnosti pre rozmiestnenie v rámci scrollovacieho okna ako sú vonkajšie okraje, vzájomné odd-
        # delenie a zarovnanie v rámci elementu.
        self._scroll_area_l.setContentsMargins(0, 0, 0, 0)
        self._scroll_area_l.setSpacing(2)
        self._scroll_area_l.setAlignment(QtCore.Qt.AlignTop)

        # Scrollovaciemu oknu je nastavený obsahový lement. Do scrollovacieho okna je pridaný AddRemoveComobobox, kto-
        # rému je nastavená aj minimálna výška, aby sa pri zmene veľkosti okna nedeformoval do neprirodzených
        # rozmerov.
        self._scroll_win.setWidget(self._scroll_area_c)
        self._scroll_area_l.addWidget(self._add_slider_rc)
        self._add_slider_rc.setMinimumHeight(100)

    def addSlider_visibility_test(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda testuje, či má byť zobrazený grafický prvok AddRemoveCombobox, podľa toho či už boli zobrazené všetky
        príspustné slidere.
        """
        if len(self._act_slider_dict) == self._pos_n_of_sliders:
            self._add_slider_rc.hide()
        else:
            self._add_slider_rc.show()

    def handle_combobox_input(self, item: tuple):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda je pripojená na AddRemoveCombobox a po zvolení niektorého zo sliderov je zavolaná. Na základe parametra
        je identifikovaný požadovnaý slider a nasleduje jeho vytvorenie a inicializácia.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item: usporiadaná dvojica, ktorá obsahuje údaje na identifikáciu slidera, ktorý má byť zobrazený.
        """
        # Je potrebné zistiť, aké identifikačné číslo mal prvok priradené. Ak bolo identifikačné číslo menšie ako 0,
        # bolo kliknuté na špeciálnu položku, v tomto prípade na položku Select all, ktorá ma zobraziť všetky ešte
        # nezobrazené slidery.
        if item[0] >= 0:
            # Ak bolo identifikačné číslo väčšie alebo rovné ako 0, tak používateľ chce zobraziť slider, ktorého unikát-
            # ne meno sa nachádza na druhej pozicií v rámci poskytnutého parametra. Najskôr je však potrebné zistiť, či
            # už daný slider nie je zobrazený, pretože do AddRemoveComboxu je možné písať konrétne názvy sliderov, aby
            # ich nebolo potrebné vyhľadávať v listboxe.
            if item[1] not in self._act_slider_dict.keys():
                # Ak slider ešte nebol zobrazený, testuje sa, či zobrazením ďalšieho slidera nepresiahne počet zobraze-
                # ných sliderov kritickú hodnotu. Ak by bola táto kritická hodnota prekročená, poklesol by výkon
                # aplikácie a mohla by sa aj úplne zaseknúť.
                if len(self._act_slider_dict) + 1 > 1500:
                    # V prípade ak by pridaním slidera bol prekročený kritický počet, je zobrazené varovné okno, ktoré
                    # od užívateľa požaduje informáciu, či chce aj napriek tomu pokračovať.
                    if self.show_warning() != 0:
                        # Ak si používateľ zvolí, že nechce pokračovat, metóda konćí a slider nie je zobrazený.
                        return
                # Ak sa používateľ rozhodne pokračovať je zaslaný identifikátor slidera do metódy zodpovednej za jeho
                # zobrazenie.
                self.add_slider(item[1])
        else:
            # V prípade, že bola zvolená špeciálna možnosť, sú z AddRemoveCombobox inštancie získané unikátne názvy ešte
            # nezobrazených sliderov.
            list_of_remaining = self._add_slider_rc.get_list_of_visible()
            # Prvé dve hodnoty z tohto listu je však potrebné preskočiť, pretože ide o možnosti placeholder a
            # Select all.
            list_of_remaining = list_of_remaining[1:].copy()
            # Je overené či pridaním daného počtu sliderov nepresiahne počet sliderov kritickú hranicu.
            if len(self._act_slider_dict) + len(list_of_remaining) > 6:
                # Ak túto hranicu presiahne, je o tom používateľ informovaný a čaká sa od neho odpoveď či chce napriek
                # tomu pokračovať.
                if self.show_warning() != 0:
                    # Ak si nezvolí možnosť, že chce pokračovať, alebo okno zavrie, metóda končí.
                    return
            # Následne je list nezobrazených sliderov postupne prechádzaný a slidere sú zobrazované.
            for name in list_of_remaining:
                self.add_slider(name)

    def show_warning(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda zobrazí varovné okno informujúce používateľa o tom, že pridaním ďalšieho slidera môže spôsobiť problémy
        s fungovaním aplikácie.

        Návratová hodnota:
        ----------------------------------------------------------------------------------------------------------------
        Vracia číslo zodpovedajúce tlačidlu, na ktoré bolo kliknuté.
        """
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

    def add_slider(self, slider_name: str):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Ide o metódu, v rámci ktorej je na pomocou unikátneho mena slidera zistené, aký druh slidera má byť vytvorený.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_name: unikátne meno slidera
        """
        # Na zákalde unikátneho mena je získaná konfiguračná n-tica pre daný typ slidera, v závislosti od ktorej bude
        # ďalej zavolaná metóda, zodpovedná za vytvorenie slidera pre zmenu konrkétnych hodnôt.
        slider_config = self._slider_dict[slider_name]
        if slider_config[0]:
            self.create_weight_slider(slider_name, slider_config)
        else:
            self.create_bias_slider(slider_name, slider_config)

        # Názov slidera je odstranený listu prvkov AddRemoveCombobox.
        self._add_slider_rc.hide_item(slider_name)

    def create_weight_slider(self, slider_name, slider_config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Abstraktná metóda. V potomkoch vytvorí a nakonfiguruje váhový slider na zákalde poskytnutých parametrov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_name:     meno slidera, ktorý má byť zobrazený.
        :param slider_config:   konfigurácia, ktorá ma byť použitá pri vytváraní slidera.
        """
        pass

    def create_bias_slider(self, slider_name, slider_config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Abstraktná metóda. V potomkoch triedy, zodpovedá za vytvorenie a nakonfigurovanie bias slidera s použitím
        poskytnutých parametrov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_name:     meno slidera, ktorý má byť zobrazený.
        :param slider_config:   konfigurácia, ktorá ma byť použitá pri vytváraní slidera.
        """
        pass

    def remove_slider(self, slider_id: str):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda podľa poskytnutého identifikátora zmaže požadovaný slider a pridá ho opäť medzi nezobrazené slidere v
        prvku AddRemoveCombobox.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_id: identifikátor slidera, ktorý má byť skrytý.
        """
        slider = self._act_slider_dict.pop(slider_id)
        slider.clear()
        self._add_slider_rc.show_item(slider_id)
        self.addSlider_visibility_test()

    def on_slider_change(self, value):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda je volaná pri zmene niektorého zo zobrazených sliderov a slúži na to, aby signalizovala zmenu váh nadra-
        denej triede.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param value: parameter sa nepoužíva, no je však definovanie je vyžadované.
        """
        self._master.weight_bias_change_signal()

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vyčistí zobrazené slidere a uvoľní pridelené prostriedky.
        """
        for slider in self._act_slider_dict.values():
            slider.clear()
        self._act_slider_dict = {}
        self._add_slider_rc.clear()
        self._add_slider_rc = None

    def __del__(self):
        print('Mazanie controller')


class NoFMWeightControllerFrame(LayerWeightControllerFrame):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Ide o potomka triedy WeightControllerFrame, ktorý je vytvorený v prípade, ak majú byť ovládané vahy husto prepojenej
    neurónovej vrstvy.
    """
    def initialize(self, controller, layer_weights: list, layer_bias: list):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda v rámci ktorej sú inicializované atribúty.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param controller:    referencia na nadradenú triedu.
        :param layer_weights: referencia na pole váh vrstvy, ktoré majú byť pomocou inštancie triedy menené.
        :param layer_bias:    referencia na pole bias hodnôt vrstvy, ktoré majú byť pomocou inštancie triedy menené.
        """
        # V prípade, ak už boli vytvorené a zobrazené slidre, sú tieto vyčistené a sú vyčistené aj slovníky, ktoré
        # ktoré držali odkazy a informácie na konfiguráciu týchto sliderov.
        for weight_slider in self._act_slider_dict.values():
            weight_slider.clear()
        self._slider_dict = {}
        self._act_slider_dict = {}

        # Sú priradené referencie na nadradenú triedu a polia váh a biasov, ktoré majú byť ovladané
        self._master = controller
        self._bias_ref = layer_bias
        self._weights_ref = layer_weights

        tmp_ordered_sliders_names = []
        tmp_ordered_sliders_config = []

        if layer_weights is not None:
            # Ak existuje referencia na pole váh su na základe rozmerov poľa vytvorené unikátne mená sliderov. Tieto
            # sú vložené do pomocných premenných. Súčasne je vytvorená aj konfiguračná usporiadaná trojica, ktorá
            # obsahuje informácie pre vytvorienie slidera. Tieto informácie majú nasledujúci tvar:
            # 1. bool hodnota hovoriaca o tom, či sa jedná o slider: váh - True, biasy - False
            # 2. index začiatočného neurónu, označujúci aj príslušny riadok v poli váh
            # 3. index koncového neurónu, ozačujúci aj príslušný stĺpec v poli váh
            # Poradie prvkov, v ktorom sú do zoznamov vkladané je dôležitý.
            for start_neuron in range(len(self._weights_ref)):
                for end_neuron in range(len(self._weights_ref[start_neuron])):
                    layer_name = 'Weight {}-{}'.format(start_neuron, end_neuron)
                    tmp_ordered_sliders_names.append(layer_name)
                    tmp_ordered_sliders_config.append((True, start_neuron, end_neuron))

        if layer_bias is not None:
            # Obdobne ako pri váhach, ak je priradená referencia na pole biasov, je na základe jeho rozmeru vytvorený
            # identifikátor pre príslušny slider. Taktiež je inicializovaná aj konfigurácia s potrebnými údajmi
            # pre budúce vytvorenie slidera.
            for end_neuron in range(len(layer_bias)):
                layer_name = 'Bias {}'.format(end_neuron)
                tmp_ordered_sliders_names.append(layer_name)
                tmp_ordered_sliders_config.append((False, end_neuron))

        # Podľa počtu unikátnych mien v zozname je zaznamenaný maximálny možný počet sliderov, ktoré je možné zobraziť.
        self._pos_n_of_sliders = len(tmp_ordered_sliders_names)

        # Tento zoznam je poslaný do inicializačnej metódy triedy AddRemoveCombobox, ktorej výstupom je zaručene uni-
        # kátny zoznam mien sliderov.
        final_name_list = self._add_slider_rc.initialize(tmp_ordered_sliders_names, self.handle_combobox_input,
                                                         'Add weight', False, 'Select weight')

        # Do slovniká slider dict je každému slideru priradený kľúč vo forme jeho názvu a priradená hodnota konfigurá-
        # cie.
        for i, slider_name in enumerate(final_name_list):
            self._slider_dict[slider_name] = tmp_ordered_sliders_config[i]

        # Je pridaný špeciálny prvok, ktorý slúži na zobrazenie všetkých ešte nezobrazených sliderov.
        self._add_slider_rc.add_special('Show all')

        # Na záver je otestované, či má byť zobrazený AddRemoveCombobox.
        self.addSlider_visibility_test()

    # def add_slider(self, slider_name: str):
    #     """
    #     Popis
    #     ----------------------------------------------------------------------------------------------------------------
    #     Ide o metódu, v rámci ktorej je na pomocou unikátneho mena slidera zistené, aký druh slidera má byť vytvorený.
    #
    #     Parametre
    #     ----------------------------------------------------------------------------------------------------------------
    #     :param slider_name: unikátne meno slidera
    #     """
    #     # Na zákalde unikátneho mena je získaná konfiguračná n-tica pre daný typ slidera, v závislosti od ktorej bude
    #     # ďalej zavolaná metóda, zodpovedná za vytvorenie slidera pre zmenu konrkétnych hodnôt.
    #     slider_config = self._slider_dict[slider_name]
    #     if slider_config[0]:
    #         self.create_weight_slider(slider_name, slider_config)
    #     else:
    #         self.create_bias_slider(slider_name, slider_config)
    #
    #     # Názov slidera je odstranený listu prvkov AddRemoveCombobox.
    #     self._add_slider_rc.hide_item(slider_name)

    def create_weight_slider(self, slider_name, slider_config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vytvorí slider, ktorý je zodpovený za zmenu konkrétnej váhy na vrstve. Odkaz na tento slider je uložený v
        slovníku pod unikátnym kľúčom.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_name:   unikátne meno slidera.
        :param slider_config: konfiguračná n-tica pre vytvorenie slidera.
        """
        # Z konfiguračnej premennej sú získané hodnoty začiatočného a koncového neurónu, ktoré predstavujú aj indexy
        # do poľa váh. Z nich je vytvorené meno slidera, ktoré sa bude zobrazovať na grafickom prvku. Je vytvorená
        # inštancia prvku VariableDisplaySlider a je mu nastavená minimálna výška, aby pri zmene rozmerov okna
        # nedochádzalo k nečitateľnému zmenšeniu slidera. Následne je tento slider inicializovaný. Po inicia-
        # lizácií je slider vložený do rozmiestnenia obsahu scrollovacieho okna na pozíciu hneď pred
        # grafický prvok triedy AddRemoveCombobox. Odkaz na novo vytvorenú inštanciu slidera je pod
        # unikátnym menom vložený do slovníka. Na záver je otestované, či neboli zobrazené už
        # všetky možné slidre na vrstve.
        start_neuron = slider_config[1]
        end_neuron = slider_config[2]
        slider_name = 'Weight {}-{}'.format(start_neuron, end_neuron)
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider,
                          self._weights_ref[start_neuron], end_neuron)
        self._scroll_area_l.insertWidget(self._scroll_area_l.count() - 1, slider)
        self._act_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def create_bias_slider(self, slider_name, slider_config):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Vytvorí slider, ktorý je zodpovený za zmenu konkrétnej bias hodnoty na vrstve. Odkaz na tento slider je uložený
        v slovníku pod unikátnym kľúčom.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param slider_name:   unikátne meno slidera, pod ktorým bude v slovníku uložený odkaz na inštanciu slidera
        :param slider_config: konfiguračná premenná slidera, obsahujúca informácie pre jeho vytvorenie a inicializáciu.
        """
        # Z konfiguračnej premennej je získaný index kocového neurónu. Je vytvorená inštancia triedy
        # VariableDisplaySlider, ktorej je nastavená minimálna veľkosť. Následne je táto inštancia
        # inicializovaná, pomocou konfiguračnej premennej.
        end_neuron = slider_config[1]
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -10, 10, slider_name, self.on_slider_change, self.remove_slider,
                          self._bias_ref, end_neuron)

        # Novo vytvorený slider je vložený do rozmiestnenia obsahu scrollovacieho okna. Odkaz na inštanciu triedy je
        # uchovaný v slovníku, kde kľúčom je unikátne meno tohto slidera, poskytnuté ako parameter funkcie.
        self._scroll_area_l.insertWidget(self._scroll_area_l.count() - 1, slider)
        self._act_slider_dict[slider_name] = slider
        # Na záver je otestované, či nebol zobrazený maximálny možný počet sliderov.
        self.addSlider_visibility_test()


class FMWeightControllerFrame(LayerWeightControllerFrame):
    """
    Popis
    --------------------------------------------------------------------------------------------------------------------
    Ide o potomka triedy WeightControllerFrame, ktorý je vytvorený v prípade, ak majú byť ovládané vahy vrstvy, ktorá na
    svojom výstupe obsahuje feature mapy.
    """
    def initialize(self, controller, layer_weights: np.array, layer_bias: np.array, feature_map_n):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda v rámci ktorej sú inicializované atribúty.


        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param controller:        referencia na nadradenú triedu.
        :param layer_weights:     referencia na pole váh vrstvy, ktoré majú byť pomocou inštancie triedy menené.
        :param layer_bias:        referencia na pole bias hodnôt vrstvy, ktoré majú byť pomocou inštancie triedy menené.
        :param feature_map_n:     číslo feature mapy, pre ktorej váhy majú byť inicializované slidere.
        """
        # V prípade, ak už boli vytvorené a zobrazené slidre, sú tieto vyčistené a sú vyčistené aj slovníky, ktoré
        # ktoré držali odkazy a informácie na konfiguráciu týchto sliderov.
        for weight_slider in self._act_slider_dict.values():
            weight_slider.clear()

        self._slider_dict = {}
        self._act_slider_dict = {}

        # Sú priradené referencie na nadradenú triedu a polia váh a biasov, ktoré majú byť ovladané
        self._master = controller
        self._bias_ref = layer_bias
        self._weights_ref = layer_weights

        # Je zavolaná metóda, ktorá sa stará o inicializovanie konfigurácií pre vytváranie sliderov na ovládanie váh
        # zadanej feature mapy.
        self.initialize_for_fm(feature_map_n)

    def initialize_for_fm(self, feature_map_n):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vytvorí konfigurácie pre slidre v závislosti od požadovanej feature mapy.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param feature_map_n: číslo feature mapy, podľa ktorej majú byť slovníky obsahujúce konfiguráciu naplnené.
        """
        # V prípade, ak už boli vytvorené a zobrazené slidre, sú tieto vyčistené a sú vyčistené aj slovníky, ktoré
        # ktoré držali odkazy a informácie na konfiguráciu týchto sliderov.
        for weight_slider in self._act_slider_dict.values():
            weight_slider.clear()

        self._slider_dict = {}
        self._act_slider_dict = {}

        # Sú definované pomocné zoznami.
        tmp_ordered_sliders_names = []
        tmp_ordered_configuration = []

        if self._weights_ref is not None:
            # Ak je poskytnutý referencia na filtre neurónovej vrstvy, je pomocou nich vytvorený unikátny identifikátor
            # ktorý je vložený do pomocných zoznamov. Tiež je pre príslušne slidere definovaná konfigurácia potrebná
            # pre ich vytvorenie a inicializáciu.
            weight_shape = self._weights_ref.shape[:-1]
            for channel_i in range(weight_shape[2]):
                for row_i in range(weight_shape[0]):
                    for col_i in range(weight_shape[1]):
                        key = f'Channel{channel_i}: Weight {row_i}-{col_i}'
                        tmp_ordered_sliders_names.append(key)
                        tmp_ordered_configuration.append([True, feature_map_n, channel_i, row_i, col_i])
        if self._bias_ref is not None:
            # Obdobne ako pri váhach, ak je priradená referencia na pole biasov, je na základe jeho rozmeru vytvorený
            # identifikátor pre príslušny slider. Taktiež je inicializovaná aj konfigurácia s potrebnými údajmi
            # pre budúce vytvorenie slidera.
            tmp_ordered_sliders_names.append(f'Bias{feature_map_n}')
            tmp_ordered_configuration.append([False, feature_map_n])

        # Pomocou pomocných zoznamov je inicializovný AddRemoveCombobox, ktorý navráti zaručene unikátne identifikátory
        # sliderov.
        tmp_ordered_sliders_names = self._add_slider_rc.initialize(tmp_ordered_sliders_names, self.handle_combobox_input,
                                                                   'Add weight', False, 'Select weight')

        # Titeto unikátne mená sú použité ako kľúče do slovníka konfigurácií.
        for i, key in enumerate(tmp_ordered_sliders_names):
            self._slider_dict[key] = tmp_ordered_configuration[i]

        # Na základe dĺžky zoznamu unikátnych mien je určený maximálny počet možných sliderov.
        self._pos_n_of_sliders = len(tmp_ordered_sliders_names)

        # Na záver je otestované, či má byť zobrazený AddRemoveCombobox.
        self.addSlider_visibility_test()
        self._add_slider_rc.add_special('Show all')

    # def add_slider(self, slider_name: str):
    #     slider_config = self._slider_dict[slider_name]
    #     if slider_config[0]:
    #         self.create_weight_slider(slider_name, slider_config)
    #     else:
    #         self.create_bias_slider(slider_name, slider_config)
    #     self._add_slider_rc.hide_item(slider_name)

    def create_weight_slider(self, slider_name, slider_config):
        feature_map = self._weights_ref[:, :, :, slider_config[1]]
        channel = feature_map[:, :, slider_config[2]]
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -1, 1, slider_name, self.on_slider_change, self.remove_slider,
                          channel[slider_config[3]], slider_config[4])
        self._scroll_area_l.insertWidget(self._scroll_area_l.count() - 1, slider)
        self._act_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

    def create_bias_slider(self, slider_name, slider_config):
        slider = VariableDisplaySlider()
        slider.setMinimumHeight(60)
        slider.initialize(slider_name, -10, 10, slider_name, self.on_slider_change, self.remove_slider,
                          self._bias_ref, slider_config[1])
        self._scroll_area_l.insertWidget(self._scroll_area_l.count() - 1, slider)
        self._act_slider_dict[slider_name] = slider
        self.addSlider_visibility_test()

