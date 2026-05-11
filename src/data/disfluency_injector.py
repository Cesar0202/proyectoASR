import random

class DisfluencyInjector:
    # Esta clase mete los errores en el texto
    def aplicar(self, texto, tipo, severidad, intensidad):
        palabras = texto.split()
        if tipo == "repetition":
            return self._repetir(palabras, severidad, intensidad)
        elif tipo == "prolongation":
            return self._alargar(palabras, severidad, intensidad)
        elif tipo == "filler":
            return self._muletilla(palabras, severidad, intensidad)
        elif tipo == "block":
            return self._pausa(palabras, severidad, intensidad)
        elif tipo == "mix":
            return self._mezcla(palabras, severidad, intensidad)
        return texto

    def _repetir(self, palabras, sev, inten):
        res = []
        for p in palabras:
            if random.random() < sev:
                # Repetir la palabra varias veces
                veces = int(inten * 5) + 1
                for _ in range(veces):
                    res.append(p)
            else:
                res.append(p)
        return " ".join(res)

    def _alargar(self, palabras, sev, inten):
        res = []
        for p in palabras:
            if random.random() < sev and len(p) > 3:
                # Alargar la primera letra
                letra = p[0]
                alargue = letra * (int(inten * 10) + 2)
                res.append(alargue + "-" + p)
            else:
                res.append(p)
        return " ".join(res)

    def _muletilla(self, palabras, sev, inten):
        muletillas = ["eh...", "este...", "o sea...", "ahm...", "bueno..."]
        res = []
        for p in palabras:
            if random.random() < sev:
                res.append(random.choice(muletillas))
            res.append(p)
        return " ".join(res)

    def _pausa(self, palabras, sev, inten):
        res = []
        for p in palabras:
            if random.random() < sev:
                res.append("...")
            res.append(p)
        return " ".join(res)

    def _mezcla(self, palabras, sev, inten):
        # Hace un poco de cada cosa
        texto = " ".join(palabras)
        texto = self._repetir(texto.split(), sev/2, inten)
        texto = self._muletilla(texto.split(), sev/2, inten)
        return texto
