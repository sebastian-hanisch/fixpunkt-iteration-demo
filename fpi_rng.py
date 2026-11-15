"""Eigener Zufallsgenerator (SplitMix64), nur für die Eigenschafts- und Orakel-Tests (Zufallsmatrizen, Zufallsspektren).

Herkunft: Kopie aus `saal-abstand-demo/sab_rng.py`, auf den Umfang dieser Demo verkürzt und umbenannt. Gleiche Zufalls-Seeds geben auf jeder
Python-Version und Plattform dieselben Folgen; Python-`random` und `hash()` sind dafür nicht zugesichert. Die App selbst zieht nichts zufällig."""

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()
