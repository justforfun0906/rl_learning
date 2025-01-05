import pygame
import sys
from GameEnv import GameEnv

# === 一些常數設定 ===
GRID_SIZE = 4             # 4x4 棋盤
TILE_SIZE = 100           # 每個方塊的大小 (px)
TILE_MARGIN = 10          # 方塊間距 (px)
WINDOW_SIZE = 800      # 遊戲視窗寬度 & 高度 (可依需求調整)
BG_COLOR = (187, 173, 160)  # 背景色
EMPTY_COLOR = (205, 193, 180)

# 標題字體大小、預設字體大小
TITLE_FONT_SIZE = 40
SCORE_FONT_SIZE = 24

# 顏色配置，可依需求自行修改
TILE_COLORS = {
    0:    (205, 193, 180),
    2:    (238, 228, 218),
    4:    (237, 224, 200),
    8:    (242, 177, 121),
    16:   (245, 149, 99),
    32:   (246, 124, 95),
    64:   (246, 94, 59),
    128:  (237, 207, 114),
    256:  (237, 204, 97),
    512:  (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    # 更大的數字可以自行新增
}

# 數字字體顏色：當數字比較小時使用較深色，大數字使用淺色
FONT_COLORS = {
    2:    (119, 110, 101),
    4:    (119, 110, 101),
    8:    (249, 246, 242),
    16:   (249, 246, 242),
    32:   (249, 246, 242),
    64:   (249, 246, 242),
    128:  (249, 246, 242),
    256:  (249, 246, 242),
    512:  (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
    # 以上可根據實際需要增補
}


def draw_board(screen, board, score, font, title_font):
    """
    將整個棋盤繪製到畫面上。
    """
    screen.fill(BG_COLOR)

    # 繪製標題
    title_surface = title_font.render("2048", True, (119, 110, 101))
    screen.blit(title_surface, (10, 10))

    # 繪製分數
    score_surface = font.render(f"Score: {score}", True, (119, 110, 101))
    screen.blit(score_surface, (10, 60))

    # 繪製方塊
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            value = board[r][c]
            color = TILE_COLORS.get(value, TILE_COLORS[2048])  # 超過 2048 統一使用同色
            rect_x = c * (TILE_SIZE + TILE_MARGIN) + 10
            rect_y = r * (TILE_SIZE + TILE_MARGIN) + 110
            pygame.draw.rect(screen, color, (rect_x, rect_y, TILE_SIZE, TILE_SIZE), border_radius=5)

            if value > 0:
                text_color = FONT_COLORS.get(value, (249, 246, 242))
                tile_font = pygame.font.SysFont("arial", 32)
                text_surface = tile_font.render(str(value), True, text_color)
                text_rect = text_surface.get_rect(center=(rect_x + TILE_SIZE / 2, rect_y + TILE_SIZE / 2))
                screen.blit(text_surface, text_rect)

def random_actor():
    """
    隨機產生一個動作 (上, 下, 左, 右)
    可透過字串或數字等方式回傳
    """
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    return random.choice(actions)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("2048 with Pygame")

    clock = pygame.time.Clock()

    score_font = pygame.font.SysFont("arial", SCORE_FONT_SIZE, bold=True)
    title_font = pygame.font.SysFont("arial", TITLE_FONT_SIZE, bold=True)

    env = GameEnv()
    state = env.reset()
    score = 0

    running = True
    while running:
        clock.tick(60)
        
        # 1) 先檢查事件 (保留關閉視窗、其他事件)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2) 呼叫隨機 Actor 決定動作
        action = random_actor()

        # 3) 依照 Actor 的動作決定 move
        next_state, reward, done = env.step(action)
        score += reward

        # 4) 更新畫面
        draw_board(screen, env.board, score, score_font, title_font)
        pygame.display.flip()

        # 5) 檢查遊戲是否結束
        if done:
            print("Game Over!")
            running = False
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()