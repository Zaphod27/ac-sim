from __future__ import annotations

import abc
import dataclasses
import math
import typing


def celsius(kelvin: float) -> float:
    return kelvin - 273.15


def kelvin(celsius: float) -> float:
    return celsius + 273.15


UNIVERSAL_GAS_CONSTANT = 8.314  # J K^-1 mol^-1
MOLAR_MASS_WATER = 0.018015  # kg mol^-1
# TODO: specific density given the temp
DENSITY_AIR = 1.204  # kg m^-3


def water_vapour_pressure(temp: float) -> float:
    """Pa = N / m^2"""
    T_celsius = celsius(temp)

    return 0.61121 * math.exp((18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))) * 1000


def max_solved_water_in_air(temp: float) -> float:
    """kg m^-3"""
    P = water_vapour_pressure(temp)

    return P * MOLAR_MASS_WATER / (UNIVERSAL_GAS_CONSTANT * temp)


def max_solved_water_in_air2(temp: float):
    """kg water / kg dry air"""
    return max_solved_water_in_air(temp) / DENSITY_AIR


def water_enthalpy(temp: float):
    """J / kg"""
    ZERO_C = 2_500_900  # J / kg
    HUNDRED_C = 2_256_400  # J / kg

    temp_c = celsius(temp)

    diff_per_c = (HUNDRED_C - ZERO_C) / 100

    return ZERO_C + temp_c * diff_per_c


@dataclasses.dataclass(frozen=True)
class MatterState(abc.ABC):
    temp: float  # K

    air: float  # kg
    water_vapor: float  # kg
    water: float  # kg
    desiccant: float  # kg

    def with_(self, **kwargs) -> typing.Self:
        # noinspection PyArgumentList
        return self.__class__(
            **(dataclasses.asdict(self) | kwargs)
        )

    def with_energy(self, add_energy: float) -> typing.Self:
        return self.with_(
            temp=(self.heat_capacity() * self.temp + add_energy) / self.heat_capacity()
        )

    def thermal_energy(self):
        return self.heat_capacity() * self.temp

    def mass(self) -> float:
        return self.air + self.water_vapor + self.water + self.desiccant

    def air_volume(self):
        return self.air * DENSITY_AIR

    def heat_capacity(self):
        """J K^-1"""
        # H = self.solved_water / DENSITY_AIR
        # return (1.005 + 1.82 * H) * 1000
        air = 1_003.5 * self.air
        water_vapour = 2_110 * self.water_vapor
        water = 4_186 * self.water
        desiccant = 0

        return air + water_vapour + water

    def specific_heat_capacity(self):
        """J kg^-1 K^-1"""
        return self.heat_capacity() / self.mass()

    def relative_humidity(self):
        return self.water_vapor / (max_solved_water_in_air2(self.temp) * self.air)

    def humidity_ratio(self):
        """kg water / kg dry air"""
        return self.water_vapor / self.air

    def __mul__(self, other):
        if isinstance(other, float):
            return MatterState(
                temp=self.temp,
                air=self.air * other,
                water_vapor=self.water_vapor * other,
                water=self.water * other,
                desiccant=self.desiccant * other
            )
        else:
            return NotImplemented


class Component(abc.ABC):
    inputs: tuple[str]
    outputs: tuple[str]

    @abc.abstractmethod
    def compute(self, inputs: tuple[MatterState, ...]) -> tuple[MatterState, ...]:
        pass


class Mixer(Component):
    inputs = "in1", "in2"
    outputs = "out"

    def compute(self, inputs: tuple[MatterState, MatterState]) -> tuple[MatterState]:
        in1, in2 = inputs

        a1 = in1.heat_capacity()
        a2 = in2.heat_capacity()

        T = (in1.temp * a1 + in2.temp * a2) / (a1 + a2)

        return MatterState(
            T,
            in1.air + in2.air,
            in1.water_vapor + in2.water_vapor,
            in1.water + in2.water,
            in1.desiccant + in2.desiccant
        ),


class Containment(Component):
    inputs = "in",
    outputs = "out",

    def __init__(self, contents: MatterState):
        self.contents = contents
        self.mixer = Mixer()

    def compute(self, inputs: tuple[MatterState]) -> tuple[MatterState]:
        inp, = inputs

        out = self.contents * (inp.mass() / self.contents.mass())
        self.contents *= (self.contents.mass() - inp.mass()) / self.contents.mass()

        self.contents, = self.mixer.compute((self.contents, inp))
        return out,


class HeatExchanger(Component):
    inputs = "in1", "in2"
    outputs = "out1", "out2"

    def __init__(self, efficiency: float):
        self.efficiency = efficiency

    def compute(self, inputs: tuple[MatterState, MatterState]) -> tuple[MatterState, MatterState]:
        in1, in2 = inputs

        a1 = in1.heat_capacity()
        a2 = in2.heat_capacity()

        if a1 > a2:
            out2, out1 = self.compute((in2, in1))
            return out1, out2

        delta_t_1_max = in2.temp - in1.temp

        q_max = delta_t_1_max * a1
        q = q_max * self.efficiency

        return in1.with_energy(q), in2.with_energy(-q)


class Sink(Component):
    inputs = "in",
    outputs = ()

    def __init__(self):
        self.contents = MatterState(0, 0, 0, 0, 0)

    def compute(self, inputs: tuple[MatterState]) -> tuple[()]:
        in1, = inputs

        self.contents = Mixer().compute((self.contents, in1))

        return ()


class Source(Component):
    inputs = ()
    outputs = "out"

    def __init__(self, matter_state: MatterState):
        self.matter_state = matter_state

    def compute(self, inputs: tuple[()]) -> tuple[MatterState]:
        return self.matter_state,


class Sprinkler(Component):
    inputs = "liquid_in", "air_in"
    outputs = "liquid_out", "air_out"

    def __init__(self, diameter: float, surface_area: float):
        self.diameter = diameter  # m
        self.surface_area = surface_area  # m^2

        self.mixer = Mixer()

    @property
    def cross_sectional_area(self):
        """m^2"""
        return math.pi * (self.diameter / 2) ** 2

    def compute(self, inputs: tuple[MatterState, MatterState]) -> tuple[MatterState, MatterState]:
        liquid_in, air_in = inputs

        velocity_air = air_in.air_volume() / self.cross_sectional_area  # m / s

        evaporation_coefficient = (25 + 19 * velocity_air) / 3600  # kg m^-2 s^-1

        mass_evaporated_water = (
                evaporation_coefficient
                * self.surface_area
                * (max_solved_water_in_air2(liquid_in.temp) - air_in.humidity_ratio())
        )  # kg

        heat_supply = water_enthalpy(liquid_in.temp) * mass_evaporated_water  # J

        assert heat_supply > 0

        evaporated_water = MatterState(
            temp=liquid_in.temp,
            air=0,
            water_vapor=mass_evaporated_water,
            water=0,
            desiccant=0
        ).with_energy(-heat_supply)

        air_out, = self.mixer.compute((air_in, evaporated_water))

        liquid_out = liquid_in.with_(
            water=liquid_in.water - mass_evaporated_water
        )

        return liquid_out, air_out


class Evaporator(Component):
    inputs = "liquid_in", "air_in"
    outputs = "liquid_out", "air_out"

    def __init__(self, heat_exchanger_1: HeatExchanger, sprinkler: Sprinkler, heat_exchanger_2: HeatExchanger):
        self.heat_exchanger1 = heat_exchanger_1
        self.sprinkler = sprinkler
        self.heat_exchanger2 = heat_exchanger_2

    @classmethod
    def create(
            cls,
            heat_exchanger_1_efficiency: float,
            sprinkler_surface_area: float,
            sprinkler_diameter: float,
            heat_exchanger_2_efficiency: float
    ):
        return cls(
            HeatExchanger(heat_exchanger_1_efficiency),
            Sprinkler(diameter=sprinkler_diameter, surface_area=sprinkler_surface_area),
            HeatExchanger(heat_exchanger_2_efficiency)
        )

    def compute(self, inputs: tuple[MatterState, MatterState]) -> tuple[MatterState, MatterState]:
        one = self.heat_exchanger1.compute(
            inputs
        )
        two = self.sprinkler.compute(
            one
        )
        three = self.heat_exchanger2.compute(
            two
        )
        return three


def main():
    one = MatterState(
        kelvin(30),
        0,
        0,
        0.500,
        0
    )
    two = MatterState(
        kelvin(30),
        0.200,
        max_solved_water_in_air2(kelvin(30)) * 0.5 * 0.200,
        0,
        0
    )
    out1, out2 = Evaporator.create(
        heat_exchanger_1_efficiency=0.50,
        sprinkler_surface_area=3,
        sprinkler_diameter=0.30,
        heat_exchanger_2_efficiency=0.80,
    ).compute((one, two))

    # one = MatterState(
    #     20,
    #     0,
    #     0,
    #     2,
    #     0
    # )
    # two = MatterState(
    #     30,
    #     0,
    #     0,
    #     1,
    #     0
    # )
    # out1, out2 = HeatExchanger(efficiency=0.5).compute((one, two))
    breakpoint()


if __name__ == "__main__":
    main()
