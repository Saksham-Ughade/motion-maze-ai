# Motion Maze AI

An AI-powered motion-controlled maze game built using Python, MediaPipe, and Pygame.  
The player is controlled using real-time hand gestures, and the maze is generated dynamically using graph traversal algorithms.

---

## Project Overview

Motion Maze AI is an interactive game that combines computer vision and pathfinding algorithms.  
The system captures hand gestures using MediaPipe and translates them into player movement within a dynamically generated maze.

The project demonstrates practical implementation of DFS, BFS, backtracking, and real-time gesture recognition.

---

## Key Features

- Real-time hand gesture control using MediaPipe
- Maze generation using Depth First Search (DFS) with backtracking
- Shortest path computation using Breadth First Search (BFS)
- Animated path visualization
- Smooth tile-based movement
- Background music and sound effects
- Multiple difficulty levels (Easy, Medium, Hard)

---

## Algorithms Used

### Depth First Search (DFS)
Used to generate a fully connected maze structure with backtracking to ensure all cells are visited.

### Breadth First Search (BFS)
Used to compute and animate the shortest path from the start position to the goal.

---

## Tech Stack

- Python
- Pygame
- OpenCV
- MediaPipe
- NumPy

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python motion_maze_fingers.py
```

---

## Project Structure

```
motion-maze-ai/
│
├── motion_maze_fingers.py
├── maze_game.py
├── finger_control.py
├── camera_test.py
├── assets/
└── requirements.txt
```

---

## Author

Saksham Ughade
