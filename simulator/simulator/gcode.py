from abc import ABC, abstractmethod
from json import tool
from os import name
from re import M
from tracemalloc import start
from typing import Dict
import cairo
from pydantic import BaseModel


TIME_RESOLUTION_MS = 5
POINT_RESOLUTION = 0.01


# Custom Exceptions
class GCodeException(Exception):
    pass


class CrashException(GCodeException):
    pass


class ValidationException(GCodeException):
    pass


# Geometric Classes
class Point2(BaseModel):
    x: float
    y: float


class Point3(Point2):
    z: float


class TargetPoint3(Point3):
    f: float

    @classmethod
    def from_gcode(cls, gcode: str) -> "TargetPoint3":
        parts = gcode[3:].strip().split(" ")
        x = y = z = f = None
        for part in parts:
            if part.startswith("X"):
                x = float(part[1:])
            elif part.startswith("Y"):
                y = float(part[1:])
            elif part.startswith("Z"):
                z = float(part[1:])
            elif part.startswith("F"):
                f = float(part[1:])
            else:
                raise ValidationException(f"Invalid Part: {part}")

        if x is None or y is None or z is None or f is None:
            raise ValidationException(f"Invalid GCODE: {gcode}")
        return cls(x=x, y=y, z=z, f=f)


# Tool Classes
class Color(BaseModel):
    r: float
    g: float
    b: float


class Tool(ABC):
    def __init__(
        self,
        color: Color,
        contact_start_z: float = 1,
        contact_max_z: float = 0,
        tool_max_radius: float = 10,
        opacity_1_a: float = 20,
        max_speed: float = 100,
    ) -> None:
        self.color = color
        self.CONTACT_START_Z = contact_start_z
        self.CONTACT_MAX_Z = contact_max_z
        self.TOOL_MAX_RADIUS = tool_max_radius
        self.OPACITY_1_A = opacity_1_a
        self.MAX_SPEED = max_speed

    @abstractmethod
    def draw_stroke(self, start: Point3, target: TargetPoint3, canvas: "Canvas") -> None:
        pass


class Pen(Tool):
    def _stroke_to(self, start: Point3, target: TargetPoint3, axy: float, canvas: "Canvas") -> None:
        start_radius = (self.CONTACT_START_Z - start.z) * self.TOOL_MAX_RADIUS
        end_radius = (self.CONTACT_START_Z - target.z) * self.TOOL_MAX_RADIUS
        opacity = min(self.OPACITY_1_A * (1 / axy), 1) if axy > 0 else 1
        canvas.stroke_to(start_radius=start_radius, end_radius=end_radius, opacity=opacity, target=target)

    def draw_stroke(self, start: Point3, target: TargetPoint3, canvas: "Canvas") -> None:
        # Prevent crashes
        if start.z < self.CONTACT_MAX_Z or target.z < self.CONTACT_MAX_Z:
            raise CrashException("Tool has crashed.")

        # Prevent over speed
        if target.f > self.MAX_SPEED:
            raise ValidationException("Speed too high.")

        # Calculate position changes
        dx = abs(target.x - start.x)
        dy = abs(target.y - start.y)
        dz = abs(target.z - start.z)
        dxyz = dx + dy + dz

        # Calculate rates
        ax = target.f * (dx / dxyz)
        ay = target.f * (dy / dxyz)
        axy = target.f * ((dx + dy) / dxyz)
        az = target.f * (dz / dxyz)

        # Stroke starts out of contact
        if start.z > self.CONTACT_START_Z:
            # Stroke ends out of contact
            if target.z >= self.CONTACT_START_Z:
                canvas.move_to(target)

            # Stroke ends in contact
            else:
                # Calculate move to conact point
                cdz = abs(self.CONTACT_START_Z - start.z)
                c_percent = cdz / dz

                contact_target = TargetPoint3(
                    x=start.x + c_percent * dx, y=start.y + c_percent * dy, z=self.CONTACT_START_Z, f=target.f
                )

                canvas.move_to(contact_target)

                # Calculate Stroke after contact - math assumes CONTACT_MAX_Z = 0
                self._stroke_to(contact_target, target, axy, canvas)

        # Stroke starts in contact
        else:
            # Stroke ends in contact
            if target.z <= self.CONTACT_START_Z:  # TODO: Should this be <= or < ?
                self._stroke_to(start, target, axy, canvas)

            # Stroke ends out of contact
            else:
                cdz = abs(start.z - self.CONTACT_START_Z)
                c_percent = cdz / dz

                contact_target = TargetPoint3(
                    x=start.x + c_percent * dx, y=start.y + c_percent * dy, z=self.CONTACT_START_Z, f=target.f
                )

                self._stroke_to(start, contact_target, axy, canvas)

                canvas.move_to(target)


class Canvas:
    def __init__(
        self,
        width: int = 1000,
        height: int = 1000,
        tools: Dict[str, Tool] = {
            "M01": Pen(color=Color(r=1, g=0, b=0)),
            "M02": Pen(color=Color(r=0, g=1, b=0)),
            "M03": Pen(color=Color(r=0, g=0, b=1)),
        },
        tool_change_time=5000,
    ) -> None:
        self.width = width
        self.height = height
        self.tools = tools
        self.tool_change_time = tool_change_time
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.context = cairo.Context(self.surface)
        self.context.set_source_rgb(1, 1, 1)
        self.context.rectangle(0, 0, width, height)
        self.context.fill()
        self.time = 0
        self.tool = None
        self.position = Point3(x=0, y=0, z=0)

    def parse_gcode(self, gcode: str) -> None:
        lines = gcode.split("\n")
        for line in lines:
            self.parse_line(line)

    def parse_line(self, line: str) -> None:
        line = line.strip()
        print(f"Executing: {line}")
        if line == "":  # Skip empty lines
            return
        elif line.startswith("G01"):
            self.execute_feed(line)
        elif line.startswith("M0"):
            self.execute_tool_change(line)
        else:
            raise ValidationException(f"Command not supported: {line}")

    def execute_tool_change(self, line: str) -> None:
        tool_id = line.strip()
        if tool_id not in self.tools:
            raise ValidationException(f"Tool not found: {tool_id}")
        tool = self.tools[tool_id]
        print("   Tool change:", tool)
        self.tool = tool
        self.time += self.tool_change_time

    def execute_feed(self, line: str) -> None:
        if self.tool is None:
            raise ValidationException("No tool selected.")
        target = TargetPoint3.from_gcode(line)
        self.tool.draw_stroke(self.position, target, self)

    def stroke_to(self, start_radius: float, end_radius: float, opacity: float, target: TargetPoint3) -> None:
        print("   Stroke to:", target, "With opacity:", opacity)
        # Draw stroke - super approximate
        tool: Tool = self.tool  # type: ignore
        self.context.set_source_rgba(tool.color.r, tool.color.g, tool.color.b, opacity)
        self.context.set_line_width((start_radius + end_radius) / 2)
        self.context.move_to(self.position.x, self.position.y)
        self.context.line_to(target.x, target.y)
        self.context.set_line_cap(cairo.LINE_CAP_ROUND)
        self.context.stroke()

        # Update time and position
        self.time += (
            abs(target.x - self.position.x)
            + abs(target.y - self.position.y)
            + abs(target.z - self.position.z) / target.f
        )
        self.position = target

    def move_to(self, target: TargetPoint3) -> None:
        print("   Move to:", target)
        # Move
        self.context.move_to(target.x, target.y)

        # Update time and position
        self.time += (
            abs(target.x - self.position.x)
            + abs(target.y - self.position.y)
            + abs(target.z - self.position.z) / target.f
        )
        self.position = target

    def render_canvas(self) -> None:
        print("Rendering Canvas after", self.time, "ms")
        self.surface.write_to_png("example.png")


if __name__ == "__main__":
    canvas = Canvas(width=500, height=500)
    canvas.parse_gcode(
        """
        M01
        G01 X0 Y0 Z2 F100
        G01 X20 Y20 Z2 F50
        G01 X20 Y20 Z0.5 F50
        G01 X20 Y200 Z0.1 F50
        G01 X200 Y200 Z0.9 F50
        M02
        G01 X200 Y20 Z0.1 F40
        G01 X20 Y20 Z0.5 F20
        M03
        G01 X200 Y200 Z0.8 F100
        """
    )
    canvas.render_canvas()
