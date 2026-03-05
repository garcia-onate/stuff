"""Rendering components for TripOpt Gym visualization.

This module contains the scrolling track display and data grid rendering
used by the environment.
"""

import pygame
from pygame import gfxdraw
import pandas as pd
import numpy as np


class RollingMap:
    """Scrolling track visualization with elevation, speed limits, and train position."""
    
    def __init__(self, route_data):
        """Initialize the rolling map display.
        
        Parameters
        ----------
        route_data : pd.DataFrame
            Route data containing Distance, Elevation, Speed Limit columns
        """
        self.route_data = route_data
        self.speed_data = pd.DataFrame(columns=['distance', 'speed'])

        # The Map is broken into 4 sections (Title, Scoreboard, Track Data and Speed Data)
        self.mapHeight = 272
        self.titleHeight = 32
        self.scoreBoardHeight = 96
        self.speedAndTrackHeight = 144
        self.halfHeight = (self.speedAndTrackHeight / 2)
        self.mapWidth = 720

        self.TRACK_PROFILE_NUM_GRAPH_POINTS = self.mapWidth
        self.TRACK_PROFILE_MILES_IN_GRAPH = 7

        # Calculate elevation bounds from actual data
        self.minElevation = route_data['Elevation'].min()
        self.maxElevation = route_data['Elevation'].max()
        # Add some padding to prevent edge cases
        elevation_range = self.maxElevation - self.minElevation
        if elevation_range < 1:  # Handle flat routes
            elevation_range = 1
        self.minElevation -= elevation_range * 0.05
        self.maxElevation += elevation_range * 0.05
        
        self.ELEVATION_VERTICAL_SCALE = 0.8
        self.ELEVATION_BUMP = 0.1
        self.lowerMapBottomY = self.mapHeight - self.scoreBoardHeight - self.titleHeight

        self.trainLength = 2
        self.milesAfterLoco = max(2, ((self.trainLength + 0.5) * 2) / 2)

    def update(self, pygame, loco_location, loco_spd):
        """Update and render the map.
        
        Parameters
        ----------
        pygame : module
            Pygame module
        loco_location : float
            Current locomotive location in miles
        loco_spd : float
            Current locomotive speed in mph
            
        Returns
        -------
        pygame.Surface
            Rendered canvas
        """
        self.locoLocation = loco_location
        self.speed_data.loc[len(self.speed_data.index)] = [loco_location, loco_spd]

        canvas = pygame.Surface((self.mapWidth, self.mapHeight))
        canvas.fill((0, 0, 0))

        polySpdLim = []
        for index, row in self.route_data.iterrows():
            if index > 0:
                if self.route_data.loc[index]['Effective Speed Limit'] != self.route_data.loc[index - 1]['Effective Speed Limit']:
                    polySpdLim.append(tuple((self.pointLoc(self.route_data.loc[index - 1]['Distance In Route']), self.spdLoc(self.route_data.loc[index]['Effective Speed Limit']))))

            polySpdLim.append(tuple((self.pointLoc(row['Distance In Route']), self.spdLoc(row['Effective Speed Limit']))))

        polyLocoSpd = []
        for index, row in self.speed_data.iterrows():
            polyLocoSpd.append(tuple((self.pointLoc(row['distance']), self.spdLoc(row['speed']))))

        polyElevation = []
        polyElevation.append(tuple((0, self.lowerMapBottomY)))
        for index, row in self.route_data.iterrows():
            polyElevation.append(tuple((self.pointLoc(row['Distance In Route']), self.elevLoc(row['Elevation']))))

        polyElevation.append(tuple((self.pointLoc(self.route_data.loc[self.route_data.tail(1).index.item()]['Distance In Route']), self.lowerMapBottomY)))

        polyTrain = []
        polyTrain.append(tuple((self.pointLoc(self.locoLocation - self.trainLength), self.elevLoc(self.elevAtDir(self.locoLocation - self.trainLength)) - 10)))
        for index, row in self.route_data.iterrows():
            if self.locoLocation > row['Distance In Route'] > self.locoLocation - self.trainLength:
                polyTrain.append(tuple((self.pointLoc(row['Distance In Route']), self.elevLoc(row['Elevation']) - 10)))
        polyTrain.append(tuple((self.pointLoc(self.locoLocation), self.elevLoc(self.elevAtDir(self.locoLocation)) - 10)))
        polyTrain.append(tuple((self.pointLoc(self.locoLocation), self.elevLoc(self.elevAtDir(self.locoLocation)) - 1)))
        for index, row in self.route_data.loc[::-1].iterrows():
            if self.locoLocation - self.trainLength < row['Distance In Route'] < self.locoLocation:
                polyTrain.append(tuple((self.pointLoc(row['Distance In Route']), self.elevLoc(row['Elevation']) - 1)))
        polyTrain.append(tuple((self.pointLoc(self.locoLocation - self.trainLength), self.elevLoc(self.elevAtDir(self.locoLocation - self.trainLength)) - 1)))

        pygame.draw.lines(
            canvas, (255, 0, 0), points=polySpdLim, width=2, closed=False
        )

        if len(polyLocoSpd) > 1:
            pygame.draw.lines(
                canvas, (255, 255, 255), points=polyLocoSpd, width=2, closed=False
            )

        self.draw_polygon(
            canvas, polyElevation, (0, 255, 255)
        )

        self.draw_polygon(
            canvas, polyTrain, (255, 255, 255)
        )

        return canvas

    def reset(self):
        """Reset the speed data history."""
        self.speed_data = pd.DataFrame(columns=['distance', 'speed'])

    def pointLoc(self, value):
        """Convert distance to x-coordinate on canvas."""
        return (value - self.locoLocation + self.milesAfterLoco) * (self.TRACK_PROFILE_NUM_GRAPH_POINTS / self.TRACK_PROFILE_MILES_IN_GRAPH)

    def spdLoc(self, value):
        """Convert speed to y-coordinate on canvas."""
        return (70 - value) * (self.halfHeight / 70)

    def elevLoc(self, value):
        """Convert elevation to y-coordinate on canvas."""
        return self.lowerMapBottomY - (((((value - self.minElevation) / (self.maxElevation - self.minElevation)) * self.ELEVATION_VERTICAL_SCALE) + self.ELEVATION_BUMP) * self.halfHeight)

    def elevAtDir(self, value):
        """Get elevation at a specific distance via interpolation."""
        for index, row in self.route_data.iterrows():
            if row['Distance In Route'] > value:
                if index > 0:
                    x = [self.route_data.loc[index - 1]['Distance In Route'], self.route_data.loc[index]['Distance In Route']]
                    y = [self.route_data.loc[index - 1]['Elevation'], self.route_data.loc[index]['Elevation']]
                    return np.interp(value, x, y)

        return self.route_data.iloc[0]['Elevation']

    def draw_polygon(self, surface, points, color):
        """Draw anti-aliased filled polygon."""
        gfxdraw.aapolygon(surface, points, color)
        gfxdraw.filled_polygon(surface, points, color)


class DataGridView:
    """Data grid overlay for displaying train metrics."""
    
    def __init__(self, _pygame):
        """Initialize the data grid view.
        
        Parameters
        ----------
        _pygame : module
            Pygame module
        """
        self.pygame = _pygame
        self.grid_offset = 144
        self.grid_height = 96
        self.grid_width = 720
        self.grid_rows = 2
        self.grid_columns = 5
        self.label_font = _pygame.font.SysFont('calibri', 14)
        self.value_font = _pygame.font.SysFont('calibri', 14)
        self.label_color = (210, 130, 0)
        self.value_color = (0, 215, 215)
        self.grid_color = (128, 128, 128)
        self.grid_line_weight = 2

    def DrawGrid(self, canvas):
        """Draw the grid lines on the canvas.
        
        Parameters
        ----------
        canvas : pygame.Surface
            Canvas to draw on
        """
        for i in range(self.grid_rows):
            polyline = []
            polyline.append(tuple((0, self.grid_offset + (i*self.grid_height/self.grid_rows))))
            polyline.append(tuple((self.grid_width, self.grid_offset + (i*self.grid_height/self.grid_rows))))
            pygame.draw.lines(canvas, self.grid_color, points=polyline, width=self.grid_line_weight, closed=False)

        polyline = []
        polyline.append(tuple((0, self.grid_offset + (self.grid_height - self.grid_line_weight))))
        polyline.append(tuple((self.grid_width, self.grid_offset + (self.grid_height - self.grid_line_weight))))
        pygame.draw.lines(canvas, self.grid_color, points=polyline, width=self.grid_line_weight, closed=False)

        for i in range(self.grid_columns):
            polyline = []
            polyline.append(tuple((i*self.grid_width/self.grid_columns, self.grid_offset)))
            polyline.append(tuple((i*self.grid_width/self.grid_columns, self.grid_offset + self.grid_height - self.grid_line_weight)))
            pygame.draw.lines(canvas, self.grid_color, points=polyline, width=self.grid_line_weight, closed=False)

        polyline = []
        polyline.append(tuple((self.grid_width - self.grid_line_weight, self.grid_offset)))
        polyline.append(tuple((self.grid_width - self.grid_line_weight, self.grid_offset + self.grid_height - self.grid_line_weight)))
        pygame.draw.lines(canvas, self.grid_color, points=polyline, width=self.grid_line_weight, closed=False)

    def DrawValue(self, label, row, column, value, canvas):
        """Draw a labeled value in a grid cell.
        
        Parameters
        ----------
        label : str
            Label text
        row : int
            Row number (1-indexed)
        column : int
            Column number (1-indexed)
        value : float
            Value to display
        canvas : pygame.Surface
            Canvas to draw on
        """
        cell_width = (self.grid_width / self.grid_columns)
        cell_height = (self.grid_height / self.grid_rows)
        cell_x = (column - 1) * cell_width
        cell_y = (row - 1) * cell_height + self.grid_offset
        cell_center = cell_x + cell_width / 2

        value_txt = "{:.1f}".format(value)

        label_width, label_height = self.label_font.size(label)
        value_width, value_height = self.value_font.size(value_txt)

        lbl_txt_surface = self.label_font.render(label, False, self.label_color)
        lbl_rect = lbl_txt_surface.get_rect()
        lbl_rect.topleft = (cell_center - (label_width / 2), cell_y + 2)

        val_txt_surface = self.value_font.render("{:.1f}".format(value), False, self.value_color)
        val_rect = val_txt_surface.get_rect()
        val_rect.topleft = (cell_center - (value_width / 2), cell_y + label_height + 4)

        canvas.blit(lbl_txt_surface, lbl_rect)
        canvas.blit(val_txt_surface, val_rect)
