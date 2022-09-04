from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF, QObject
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF, QObject
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time

# Some global color constants that might be useful
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

# Global variable that controls the speed of the recursion automation, in seconds
#
PAUSE = 0.25

#
# This is the class you have to complete.
#


class ConvexHullSolver(QObject):

    # Class constructor
    def __init__(self):
        super().__init__()
        self.pause = False


# Some helper methods that make calls to the GUI, allowing us to send updates
# to be displayed.


    def showTangent(self, line, color):
        self.view.addLines(line, color)
        if self.pause:
            time.sleep(PAUSE)

    def eraseTangent(self, line):
        self.view.clearLines(line)

    def blinkTangent(self, line, color):
        self.showTangent(line, color)
        self.eraseTangent(line)

    def showHull(self, polygon, color):
        self.view.addLines(polygon, color)
        if self.pause:
            time.sleep(PAUSE)

    def eraseHull(self, polygon):
        self.view.clearLines(polygon)

    def showText(self, text):
        self.view.displayStatusText(text)


# This is the method that gets called by the GUI and actually executes
# the finding of the hull


    def compute_hull(self, points, pause, view):
        self.pause = pause
        self.view = view
        assert(type(points) == list and type(points[0]) == QPointF)

        t1 = time.time()
        # TODO: SORT THE POINTS BY INCREASING X-VALUE
        sortedPoints = sorted(points, key=lambda point: point.x())

        t2 = time.time()

        t3 = time.time()
        # this is a dummy polygon of the first 3 unsorted points
        # polygon = [QLineF(points[i], points[(i+1) % 3]) for i in range(3)]
        polygon = self.convex_hull(
            self.divide_and_conquer(sortedPoints))
        # TODO: REPLACE THE LINE ABOVE WITH A CALL TO YOUR DIVIDE-AND-CONQUER CONVEX HULL SOLVER
        t4 = time.time()

        # when passing lines to the display, pass a list of QLineF objects.  Each QLineF
        # object can be created with two QPointF objects corresponding to the endpoints
        self.showHull(polygon, GREEN)
        self.showText('Time Elapsed (Convex Hull): {:3.6f} sec'.format(t4-t3))

    # Transforms the list of points into a list of QlineF objects
    # This runs in O(n) for both time and space complexity as I'm generating a new list
    # that will be used to create the lines
    def convex_hull(self, polygon):
        newPolygon = [QLineF(polygon[i], polygon[(i+1) % len(polygon)])
                      for i in range(len(polygon))]

        return newPolygon

    # This is the main helped function that splits the points list into its corresponding left and right
    # subsets, which then are computed recursively
    # Time Complexity: O(nlogn) for the recursive call and then for merging them via merge function
    # Space Complexity: O(n) as I'm generating new lists and later on merging them
    def divide_and_conquer(self, points):

        if len(points) == 1:
            return points

        L = points[:len(points)//2]
        R = points[len(points)//2:]

        leftHull = self.divide_and_conquer(L)  # O(nlogn)
        rightHull = self.divide_and_conquer(R)  # O(nlogn)

        # Base case just in case there is only 3 items, in which I'm just ordering them manuallly
        if len(points) == 3:
            mergedHull = [0, 0, 0]
            mergedHull[0] = points[0]
            slope1 = (points[1].y() - points[0].y()) / \
                (points[1].x() - points[0].x())
            slope2 = ((points[2].y() - points[0].y()) /
                      (points[2].x() - points[0].x()))

            if slope2 > slope1:
                mergedHull[1] = points[2]
                mergedHull[2] = points[1]
            else:
                mergedHull[1] = points[1]
                mergedHull[2] = points[2]
            return mergedHull

        return self.merge(leftHull, rightHull)

    # This is the main function that takes care of correctly find the necessary tangents
    # in order to merge and connect the hulls
    # Time Complexity: O(n)
    # Space Complexity: O(n)

    def merge(self, leftHull, rightHull):

        # This find the leftmost and rightmost points respectively, O(n)
        leftMostPoint = leftHull.index(
            max(leftHull, key=lambda leftPoint: leftPoint.x()))
        rightMostPoint = rightHull.index(
            min(rightHull, key=lambda rightPoint: rightPoint.x()))

        # The upper common tangent can be found by scanning around the left hull in a
        # counter-clockwise direction and around the right hull in a clockwise direction.

        # Locate upper tangent, O(n)
        upperTangent = self.locate_upper_tangent(
            leftMostPoint, rightMostPoint, leftHull, rightHull)

        # Locate lower tangent, O(n)
        lowerTangent = self.locate_lower_tangent(
            leftMostPoint, rightMostPoint, leftHull, rightHull)

        # After finding the tangents, it's time to
        # This next section time complexity is O(n) and space O(n)
        mergedHull = []
        upperTangentLeft = upperTangent[0]
        upperTangentRight = upperTangent[1]
        lowerTangetLeft = lowerTangent[0]
        lowerTangetRight = lowerTangent[1]

        mergedHull.append(leftHull[lowerTangetLeft])

        while lowerTangetLeft != upperTangentLeft:
            lowerTangetLeft = (lowerTangetLeft+1) % len(leftHull)
            mergedHull.append(leftHull[lowerTangetLeft])

        mergedHull.append(rightHull[upperTangentRight])

        while upperTangentRight != lowerTangetRight:
            upperTangentRight = (upperTangentRight+1) % len(rightHull)
            mergedHull.append(rightHull[upperTangentRight])

        # if pause:
        #     self.show_recursion_button(
        #         leftHull, rightHull, upperTangent, lowerTangent)

        return mergedHull

    # Helper function to locate upper tangent - helps with debugging if the tangets were wrong
    # Creates a slope and loops over slowly decreasing or increasing slope depending on
    # which point I'm currently looking for
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    def locate_upper_tangent(self,  leftMostPoint, rightMostPoint, leftHull, rightHull):
        l, r = leftMostPoint, rightMostPoint
        slope = (rightHull[r].y() - leftHull[l].y()) / \
            (rightHull[r].x() - leftHull[l].x())

        lturned = True
        while lturned:
            lturned = False
            while True:  # keeps it moving until the slope is no longer smaller
                newSlope = (rightHull[r].y() - leftHull[(l-1) % len(leftHull)].y())/(
                    rightHull[r].x() - leftHull[(l-1) % len(leftHull)].x())
                if newSlope < slope:  # If the newSlope is decreasing, it means that I am moving to the left
                    lturned = True
                    slope = newSlope
                    l = (l-1) % len(leftHull)
                else:
                    break
            while True:  # keeps it moving until the slope is no longer bigger
                newSlope = (rightHull[(r+1) % len(rightHull)].y() - leftHull[l].y())/(
                    rightHull[(r+1) % len(rightHull)].x() - leftHull[l].x())
                if newSlope > slope:  # If the newSlope is increasing, it means that I am moving to the right
                    lturned = True
                    slope = newSlope
                    r = (r+1) % len(rightHull)
                else:
                    break

        return (l, r)

    # Helper function to locate lower tangent - helps with debugging if the tangets were wrong
    # Creates a slope and loops over slowly decreasing or increasing slope depending on
    # which point I'm currently looking for
    # Time Complexity: O(n)
    # Space Complexity: O(1)
    def locate_lower_tangent(self, leftMostPoint, rightMostPoint, leftHull, rightHull):
        l, r = leftMostPoint, rightMostPoint
        slope = (rightHull[r].y() - leftHull[l].y()) / \
            (rightHull[r].x() - leftHull[l].x())

        rturned = True
        while rturned:
            rturned = False
            while True:  # keeps it moving until the slope is no longer smaller
                newSlope = (rightHull[(r-1) % len(rightHull)].y() - leftHull[l].y())/(
                    rightHull[(r-1) % len(rightHull)].x() - leftHull[l].x())
                if newSlope < slope:  # If the newSlope is decreasing, it means that I am moving to the right
                    rturned = True
                    slope = newSlope
                    r = (r-1) % len(rightHull)
                else:
                    break
            while True:  # keeps it moving until the slope is no longer bigger
                newSlope = (rightHull[r].y() - leftHull[(l+1) % len(leftHull)].y())/(
                    rightHull[r].x() - leftHull[(l+1) % len(leftHull)].x())
                if newSlope > slope:  # If the newSlope is increasing, it means that I am moving to the left
                    rturned = True
                    slope = newSlope
                    l = (l+1) % len(leftHull)
                else:
                    break
        return (l, r)

    def show_recursion_button(self, leftHull, rightHull, upper, lower):
        leftPrint = [QLineF(leftHull[i], leftHull[(i+1) % len(leftHull)])
                     for i in range(len(leftHull))]
        rightPrint = [QLineF(rightHull[i], rightHull[(
            i+1) % len(rightHull)]) for i in range(len(rightHull))]
        upperPrint = QLineF(leftHull[upper[0]], rightHull[upper[1]])
        lowerPrint = QLineF(leftHull[lower[0]], rightHull[lower[1]])
        self.showHull(leftPrint, RED)
        self.showHull(rightPrint, PURPLE)
        self.showTangent([upperPrint, lowerPrint], BLUE)
        self.eraseHull(leftPrint)
        self.eraseHull(rightPrint)
        self.eraseTangent([upperPrint, lowerPrint])
