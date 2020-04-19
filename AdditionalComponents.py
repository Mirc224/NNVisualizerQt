import math
import queue


class Polygon:
    def __init__(self, lower_list, upper_list, divide_list=[2, 2, 2]):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Trieda slúžiaca na vytvorenie priestrorovej mriežky. Trieda definuje súradnice vrcholov a hrany tejto mriežky.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.Peaks: obsahuje v sebe zoznamy držiace súradnice vrcholov na jednotlivých osiach X, Y prípadne Z.
        :var self.Edges: zoznam hrán. Je tvorený dvojicami inexov na začiatočný a koncový vrchol v poli vrcholov.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param lower_list:  zoznam najnižších hodnôt súradnic pre jednotlivé osi.
        :param upper_list:  zoznam najvyšších hodnôt súradníc pre jednotlivé osi.
        :param divide_list: zoznam obsahujúci, na koľko častí majú byť hrany v jednotlivých smeroch rozdelené.
        """
        # Podľa minimálneho počtu zadaných súradníc v zoznamoch poskytnutých ako parameter je určený počet súradnic, čo
        # zároveň predstavuje aj dimenziu mriežky, ktorá má byť vytvorená. Do atribútu Peaks je priradený zoznam
        # obsahujúci vnorené listy, ktorých počet je rovný hodnote v premennej number_of_cords. Sú definované
        # pomocné zoznamy. Do zoznamu cord_sign bude pre každú os pridelené znamienko pomocou ktorého bude.
        # Do zoznamu max_size je uložený rozsah medzi maximálnou a minimálnou hodnotou súradnice pre každú
        # z osí.
        number_of_cords = min(len(lower_list), len(upper_list))
        self.Peaks = [[] for _ in range(number_of_cords)]
        self.Edges = []
        cord_sign = []
        max_size = []
        for i in range(number_of_cords):
            max_size.append(math.fabs(lower_list[i] - upper_list[i]))
            cord_sign.append(1 if lower_list[i] < upper_list[i] else -1)

        # Je vytvorená pomocná premenná new divide, ktorá bude držať ifnormáciu o tom, ako majú byť hrany v jednotlivých
        # smeroch rozdelené.
        new_divide = []
        if len(divide_list) != number_of_cords:
            # Najskôr je zistené či je dĺžka zoznamu totožná s počtom osí, na ktorých bude polygón vytvorený.
            number_of_divide = min(len(divide_list), number_of_cords)
            for i in range(number_of_divide):
                # Následne sa testuje, či hodnota zadaná v zozname je valídna. Ak nie, je nastavená defaultná hodnota
                # ktorá značí, že hrana nebude rozdelená.
                new_divide.append(divide_list[i])
                if new_divide[i] < 1:
                    new_divide[i] = 1
            # Ak by náhodou nebol poskytnutý dostatočný počet v zozname na delenie hrán na jednotlivých osiach, bude
            # tento zoznam palnený defaultnými hodnotami, čo znamená, že hrany na osiach nebudú delené.
            for i in range(number_of_cords - number_of_divide):
                new_divide.append(1)
        else:
            new_divide = divide_list

        # Je definoaná pomocná premenná, ktorá obsahuje offset, o ktorý sa budú posúvať vytvarané nové vrcholy v smeroch
        # osí.
        offset_list = []
        for i in range(number_of_cords):
            offset_list.append(cord_sign[i] * (max_size[i] / new_divide[i]))

        # Podľa toho, či je požadovaná 2D alebo 3D mriežka sú definované vrcholy a hrany, ktoré tieto vrcholy spájajú.
        if number_of_cords == 3:
            for z in range(new_divide[2] + 1):
                for y in range(new_divide[1] + 1):
                    for x in range(new_divide[0] + 1):
                        point_number = z * ((new_divide[1] + 1) * (new_divide[0] + 1)) + y * (new_divide[0] + 1) + x
                        self.Peaks[0].append(lower_list[0] + x * offset_list[0])
                        self.Peaks[1].append(lower_list[1] + y * offset_list[1])
                        self.Peaks[2].append(lower_list[2] + z * offset_list[2])
                        if x != new_divide[0]:
                            self.Edges.append([point_number, point_number + 1])
                        if y != new_divide[1]:
                            self.Edges.append([point_number, point_number + (new_divide[0] + 1)])
                        if z != new_divide[2]:
                            self.Edges.append([point_number, point_number + (new_divide[1] + 1) * (new_divide[0] + 1)])
        else:
            for y in range(new_divide[1] + 1):
                for x in range(new_divide[0] + 1):
                    point_number = y * (new_divide[0] + 1) + x
                    self.Peaks[0].append(lower_list[0] + x * offset_list[0])
                    self.Peaks[1].append(lower_list[1] + y * offset_list[1])
                    if x != new_divide[0]:
                        self.Edges.append([point_number, point_number + 1])
                    if y != new_divide[1]:
                        self.Edges.append([point_number, point_number + (new_divide[0] + 1)])


class QueueSet:
    def __init__(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Štruktúra, ktorá umožňuje vkladanie hodnôt bez duplicít, pričom sú pridávané do zásobníka v poradí, v ktorom
        boli vkladané.

        Atribúty
        ----------------------------------------------------------------------------------------------------------------
        :var self.queue: odkaz na štruktúru zásobník, ktorý v sebe bude držať hodnoty v poradí, ako boli vkladané.
        :var self.set:   odkaz na štruktúru set, ktorá zaručuje unikátnosť vkladaných prvkov.
        """
        self.queue = queue.Queue()
        self.set = set()

    def add(self, item):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Pomocou metódy je overené, či už prvok nie je v štruktúre prítomný. Ak nie, je prvok vložený do zásobníka a aj
        aj do štruktúry set.

        Parametre
        ----------------------------------------------------------------------------------------------------------------
        :param item: item, ktorý má byť do štruktúry vložený.
        """
        if item not in self.set:
            self.set.add(item)
            self.queue.put(item)

    def get(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vráti item, ktorý sa nachádza na vrchole zásobníka. Item je odstránený zo zásobnika a aj setu.

        """
        item = self.queue.get()
        self.set.remove(item)
        return item

    def __iter__(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda potrebná pre to, aby bola štruktúra iterovateľná.
        """
        return self

    def __next__(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda využívaná pri prechádzaní štruktúry vo for each cykle.
        """
        if len(self.set):
            return self.get()
        else:
            raise StopIteration

    def clear(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda odstráni všetky prvky zo zásobníka a setu.
        """
        self.set.clear()
        with self.queue.mutex:
            self.queue.queue.clear()

    def copy(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda na skopírovanie obsahu zásobníku. Je vytvorená pomocná inštancia QueueSet, ktorá je naplnená hodnotami.
        Takéto kopríovanie ide o deep copy triedy.
        """
        tmp = QueueSet()
        tmp.set = self.set.copy()
        for i in tmp.set:
            tmp.queue.put(i)
        return tmp

    def is_empty(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vracia informáciu o tom, ći je štruktúra prázdna.
        :return:
        """
        return False if len(self.set) else True

    def __len__(self):
        """
        Popis
        ----------------------------------------------------------------------------------------------------------------
        Metóda vracia počet prvkov v štruktúre.
        """
        return len(self.set)
