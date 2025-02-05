from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateConverter:
    x: float
    y: float

    @property
    def point_coordinates(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    conf: float
    class_id: int

    @property
    def top_left(self) -> CoordinateConverter:
        return CoordinateConverter(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> CoordinateConverter:
        return CoordinateConverter(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> CoordinateConverter:
        return CoordinateConverter(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> CoordinateConverter:
        return CoordinateConverter(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> CoordinateConverter:
        return CoordinateConverter(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float):
        return Detection(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding
        )

    def contains_point(self, point: CoordinateConverter) -> bool:
        return (self.x - self.width / 2 < point.x < self.x + self.width / 2 and
                self.y - self.height / 2 < point.y < self.y + self.height / 2)