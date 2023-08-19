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

    @classmethod
    def from_relative_humidity(
            cls,
            temp: float,
            relative_humidity: float,
            air: float = 1,
            water: float = 0,
            desiccant: float = 0
    ) -> MatterState:
        water_vapor = relative_humidity * max_solved_water_in_air2(temp) * air

        return cls(
            temp,
            air,
            water_vapor,
            water,
            desiccant
        )

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
    def compute(self, inputs: tuple[MatterState, ...], dt: float) -> tuple[MatterState, ...]:
        pass

    def get_io(self, name: str) -> tuple[typing.Self, str]:
        assert name in (self.inputs + self.outputs)
        return self, name


class Mixer(Component):
    inputs = "in1", "in2"
    outputs = "out"

    def compute(self, inputs: tuple[MatterState, MatterState], dt: float) -> tuple[MatterState]:
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

    def compute(self, inputs: tuple[MatterState], dt: float) -> tuple[MatterState]:
        inp, = inputs

        out = self.contents * (inp.mass() / self.contents.mass())
        self.contents *= (self.contents.mass() - inp.mass()) / self.contents.mass()

        self.contents, = self.mixer.compute((self.contents, inp), dt)
        return out,

    def take_out_fraction(self, fraction: float = 1):
        try:
            return self.contents * fraction
        finally:
            self.contents *= 1 - fraction

    def take_out_mass(self, amount: float):
        fraction = amount / self.contents.mass()
        return self.take_out_fraction(fraction)


class HeatExchanger(Component):
    inputs = "in1", "in2"
    outputs = "out1", "out2"

    def __init__(self, efficiency: float):
        self.efficiency = efficiency

    def compute(self, inputs: tuple[MatterState, MatterState], dt: float) -> tuple[MatterState, MatterState]:
        in1, in2 = inputs

        a1 = in1.heat_capacity()
        a2 = in2.heat_capacity()

        if a1 > a2:
            out2, out1 = self.compute((in2, in1), dt)
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

    def compute(self, inputs: tuple[MatterState], dt: float) -> tuple[()]:
        in1, = inputs

        self.contents, = Mixer().compute((self.contents, in1), dt)

        return ()


class Source(Component):
    inputs = ()
    outputs = "out",

    def __init__(self, matter_state: MatterState):
        self.matter_state = matter_state

    def compute(self, inputs: tuple[()], dt: float) -> tuple[MatterState]:
        return self.matter_state * dt,


class Probe(Component):
    inputs = "in",
    outputs = "out",

    def __init__(self):
        self.data = []
        self.times = []
        self.t = 0

    def compute(self, inputs: tuple[MatterState], dt: float) -> tuple[MatterState]:
        in1, = inputs

        self.data.append(in1)
        self.times.append(self.t)

        self.t += dt

        return in1,

    def temps(self):
        return [x.temp for x in self.data]

    def humidities(self):
        return [x.relative_humidity() for x in self.data]


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

    def compute(self, inputs: tuple[MatterState, MatterState], dt: float) -> tuple[MatterState, MatterState]:
        liquid_in, air_in = inputs

        velocity_air = air_in.air_volume() / self.cross_sectional_area / dt  # m / s

        evaporation_coefficient = (25 + 19 * velocity_air) / 3600  # kg m^-2 s^-1

        mass_evaporated_water = (
                evaporation_coefficient
                * self.surface_area
                * (max_solved_water_in_air2(liquid_in.temp) - air_in.humidity_ratio())
        )  # kg

        heat_supply = water_enthalpy(liquid_in.temp) * mass_evaporated_water  # J

        # assert heat_supply > 0

        evaporated_water = MatterState(
            temp=liquid_in.temp,
            air=0,
            water_vapor=mass_evaporated_water,
            water=0,
            desiccant=0
        ).with_energy(-heat_supply)

        air_out, = self.mixer.compute((air_in, evaporated_water), dt)

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

    def compute(self, inputs: tuple[MatterState, MatterState], dt: float) -> tuple[MatterState, MatterState]:
        one = self.heat_exchanger1.compute(
            inputs, dt
        )
        two = self.sprinkler.compute(
            one, dt
        )
        three = self.heat_exchanger2.compute(
            two, dt
        )
        return three


class Pump(Component):
    inputs = (),
    outputs = "out",

    def __init__(self, matter: MatterState):
        self.container = matter
        self.mixer = Mixer()
        self._sink = Sink()

    @property
    def sink(self):
        return self._sink

    def compute(self, inputs: tuple[()], dt: float) -> tuple[MatterState]:
        try:
            return self.container * dt,
        finally:
            self.container *= 1 - dt
            self.container, = self.mixer.compute((self.container, self.sink.contents), dt)
            self.sink.contents = MatterState(0, 0, 0, 0, 0)


class MotorizedContainment(Component):
    inputs = ()
    outputs = "out",

    def __init__(self, output_rate: float, matter: MatterState):
        self.containment = Containment(matter)
        self.output_rate = output_rate
        self._sink = Sink()

    @property
    def sink(self):
        return self._sink

    def compute(self, inputs: tuple[()], dt: float) -> tuple[MatterState]:
        sink_contents = (
            self.sink.contents
            if self.sink.contents.mass() > 0
            else self.containment.take_out_mass(self.output_rate * dt)
        )

        self.sink.contents = MatterState(0, 0, 0, 0, 0)

        return self.containment.compute((sink_contents,), dt)


class Circuit:
    def __init__(self):
        self.components: dict[int, Component] = {}
        self.out_to_in: dict[tuple[int, str], tuple[int, str] | None] = {}
        self.in_to_out: dict[tuple[int, str], tuple[int, str] | None] = {}

    def connect(self, out: tuple[Component, str], inp: tuple[Component, str]):
        out_comp, out_port = out
        in_comp, in_port = inp

        self.components[id(out_comp)] = out_comp
        self.components[id(in_comp)] = in_comp
        self.out_to_in[id(out_comp), out_port] = id(in_comp), in_port
        self.in_to_out[id(in_comp), in_port] = id(out_comp), out_port

    def compute(self, dt: float):
        new_inputs: dict[int, dict[str, MatterState | None]] = {
            id(component): {
                port: None
                for port in component.inputs
            } for component in self.components.values()
        }
        processed_components: set[int] = set()

        component_queue = [component for component in self.components.values() if len(component.inputs) == 0]

        while component_queue:
            component = component_queue.pop(0)

            inputs = tuple(
                new_inputs[id(component)][port]
                for port in component.inputs
            )

            outputs = component.compute(inputs, dt)

            for port, output in zip(component.outputs, outputs):
                next_component_in, next_component_in_port = self.out_to_in[id(component), port]

                assert next_component_in_port is not None
                new_inputs[next_component_in][next_component_in_port] = output

                has_all_inputs = all(
                    port_content is not None for port_content in new_inputs[next_component_in].values()
                )
                if has_all_inputs and next_component_in not in processed_components:
                    component_queue.append(self.components[next_component_in])

            processed_components.add(id(component))

        if len(processed_components) != len(self.components):
            raise RuntimeError("Could not process all components.")


def main():
    import matplotlib.pyplot as plt

    air_outside = MatterState.from_relative_humidity(
        temp=kelvin(28),
        relative_humidity=0.3,
    )
    air_inside = MatterState.from_relative_humidity(
        temp=kelvin(35),
        relative_humidity=0.5
    )

    heat_exchanger = HeatExchanger(efficiency=0.5)

    room = MotorizedContainment(0.500, air_inside * 40.)

    evaporator = Evaporator.create(
        heat_exchanger_1_efficiency=0.50,
        sprinkler_surface_area=3,
        sprinkler_diameter=0.30,
        heat_exchanger_2_efficiency=0.80,
    )

    outside_air_sink = Sink()
    outside_air_source = Source(air_outside * 0.2)

    probe_air_out = Probe()
    probe_room_out = Probe()

    water_tank = MotorizedContainment(0.200, MatterState(32, 0, 0, 20, 0))

    circuit = Circuit()
    circuit.connect(water_tank.get_io("out"), evaporator.get_io("liquid_in"))
    circuit.connect(evaporator.get_io("liquid_out"), water_tank.sink.get_io("in"))

    circuit.connect(outside_air_source.get_io("out"), evaporator.get_io("air_in"))
    circuit.connect(evaporator.get_io("air_out"), heat_exchanger.get_io("in1"))
    circuit.connect(heat_exchanger.get_io("out1"), outside_air_sink.get_io("in"))

    circuit.connect(room.get_io("out"), heat_exchanger.get_io("in2"))
    circuit.connect(heat_exchanger.get_io("out2"), probe_room_out.get_io("in"))
    circuit.connect(probe_room_out.get_io("out"), room.sink.get_io("in"))

    dt = 30.
    t = 0
    while t < 60 * 60 * 24 * 2:
        circuit.compute(dt)
        t += dt

    breakpoint()


if __name__ == "__main__":
    main()
