import pygame
from pygame.locals import DOUBLEBUF, HWSURFACE, QUIT, RESIZABLE, VIDEORESIZE

from onpolicy.envs.overcooked.overcooked_ai_py.utils import load_from_json


def run_static_resizeable_window(surface, fps=30):
    """
    window that can be resized and closed using gui
    """
    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(
        surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE
    )
    window.blit(surface, (0, 0))
    pygame.display.flip()
    try:
        while True:
            pygame.event.pump()
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.display.quit()
                pygame.quit()
            elif event.type == VIDEORESIZE:
                window = pygame.display.set_mode(
                    event.dict["size"], HWSURFACE | DOUBLEBUF | RESIZABLE
                )
                window.blit(
                    pygame.transform.scale(surface, event.dict["size"]), (0, 0)
                )
                pygame.display.flip()
                clock.tick(fps)
    except:
        pygame.display.quit()
        pygame.quit()
        if event.type != QUIT:  # if user meant to quit error does not matter
            raise


def vstack_surfaces(surfaces, background_color=None):
    """
    stack surfaces vertically (on y axis)
    if surfaces have different width fill remaining area with background color
    """
    result_width = max(surface.get_width() for surface in surfaces)
    result_height = sum(surface.get_height() for surface in surfaces)
    result_surface = pygame.surface.Surface((result_width, result_height))
    if background_color:
        result_surface.fill(background_color)
    next_surface_y_position = 0
    for surface in surfaces:
        result_surface.blit(surface, (0, next_surface_y_position))
        next_surface_y_position += surface.get_height()
    return result_surface


def scale_surface_by_factor(surface, scale_by_factor):
    """return scaled input surfacem (with size multiplied by scale_by_factor param)
    scales also content of the surface
    """
    unscaled_size = surface.get_size()
    scaled_size = tuple(int(dim * scale_by_factor) for dim in unscaled_size)
    return pygame.transform.scale(surface, scaled_size)


def blit_on_new_surface_of_size(surface, size, background_color=None):
    """blit surface on new surface of given size of surface (with no resize of its content), filling not covered parts of result area with background color"""
    result_surface = pygame.surface.Surface(size)
    if background_color:
        result_surface.fill(background_color)
    result_surface.blit(surface, (0, 0))
    return result_surface


class MultiFramePygameImage:
    """use to read frames of images from overcooked-demo repo easly"""

    def __init__(self, img_path, frames_path):
        self.image = pygame.image.load(img_path)
        self.frames_rectangles = MultiFramePygameImage.load_frames_rectangles(
            frames_path
        )

    def blit_on_surface(
        self, surface, top_left_pixel_position, frame_name, **kwargs
    ):
        surface.blit(
            self.image,
            top_left_pixel_position,
            area=self.frames_rectangles[frame_name],
            **kwargs
        )

    @staticmethod
    def load_frames_rectangles(json_path):
        frames_json = load_from_json(json_path)

        if (
            "textures" in frames_json.keys()
        ):  # check if its format of soups.json
            assert (
                frames_json["textures"][0]["scale"] == 1
            )  # not implemented support for scale here
            frames = frames_json["textures"][0]["frames"]

        else:  # assume its format of objects.json, terrain.json and chefs.json
            frames = []
            for filename, frame_dict in frames_json["frames"].items():
                frame_dict["filename"] = filename
                frames.append(frame_dict)

        result = {}
        for frame_dict in frames:
            assert not frame_dict.get("rotated")  # not implemented support yet
            assert not frame_dict.get("trimmed")  # not implemented support yet
            frame_name = frame_dict["filename"].split(".")[0]
            frame = frame_dict["frame"]
            rect = pygame.Rect(frame["x"], frame["y"], frame["w"], frame["h"])
            result[frame_name] = rect
        return result
