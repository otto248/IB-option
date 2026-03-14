#!/usr/bin/env python3
"""
IB SPY spread 风控脚本。

功能：
1) 获取当日持仓中的期权合约信息。
2) 实时监控 SPY 股票价格。
3) 当 SPY 价格 <= short 腿执行价时，整体仓位减半（按 spread 组合）。
4) 当 SPY 价格 <= spread 中间价（short 与 long 执行价中点）时，整体仓位平仓。

依赖：
    pip install ib_insync

运行示例：
    python ib_spy_risk_manager.py --host 127.0.0.1 --port 7497 --client-id 77
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

from ib_insync import IB, Contract, Option, Position, Stock, Ticker, util


@dataclass
class SpreadPair:
    """同到期日、同right下的一个 vertical spread 对。"""

    symbol: str
    right: str
    expiry: str
    short_leg: Position
    long_leg: Position

    @property
    def short_strike(self) -> float:
        return float(self.short_leg.contract.strike)

    @property
    def long_strike(self) -> float:
        return float(self.long_leg.contract.strike)

    @property
    def mid_strike(self) -> float:
        return (self.short_strike + self.long_strike) / 2.0

    @property
    def spread_size(self) -> int:
        return int(min(abs(self.short_leg.position), abs(self.long_leg.position)))


def today_ib_expiry_str() -> str:
    """IB 期权到期格式 yyyymmdd。"""
    return date.today().strftime("%Y%m%d")


def is_spy_option(c: Contract) -> bool:
    return isinstance(c, Option) and c.symbol == "SPY"


def load_today_spy_option_positions(ib: IB) -> List[Position]:
    """读取账户当日到期的 SPY 期权持仓。"""
    today = today_ib_expiry_str()
    positions = ib.positions()
    return [p for p in positions if is_spy_option(p.contract) and p.contract.lastTradeDateOrContractMonth == today and p.position != 0]


def build_vertical_spreads(positions: List[Position]) -> List[SpreadPair]:
    """从当日期权持仓中匹配 vertical spread（1 short + 1 long）。

    规则：
    - 同 symbol/right/expiry 分组
    - 一个 short 腿（持仓<0）匹配一个 long 腿（持仓>0）
    - 对 call spread：short strike < long strike
    - 对 put spread：short strike > long strike
    """
    grouped: Dict[Tuple[str, str, str], List[Position]] = {}
    for p in positions:
        c = p.contract
        key = (c.symbol, c.right, c.lastTradeDateOrContractMonth)
        grouped.setdefault(key, []).append(p)

    spreads: List[SpreadPair] = []
    for (symbol, right, expiry), group in grouped.items():
        shorts = [p for p in group if p.position < 0]
        longs = [p for p in group if p.position > 0]

        # 按执行价排序，便于近邻匹配
        shorts.sort(key=lambda p: float(p.contract.strike))
        longs.sort(key=lambda p: float(p.contract.strike))

        used_long = set()
        for s in shorts:
            s_strike = float(s.contract.strike)
            candidate_idx: Optional[int] = None
            best_dist = math.inf

            for i, l in enumerate(longs):
                if i in used_long:
                    continue
                l_strike = float(l.contract.strike)

                valid = (right == "C" and s_strike < l_strike) or (right == "P" and s_strike > l_strike)
                if not valid:
                    continue

                dist = abs(l_strike - s_strike)
                if dist < best_dist:
                    best_dist = dist
                    candidate_idx = i

            if candidate_idx is not None:
                used_long.add(candidate_idx)
                spreads.append(
                    SpreadPair(
                        symbol=symbol,
                        right=right,
                        expiry=expiry,
                        short_leg=s,
                        long_leg=longs[candidate_idx],
                    )
                )

    return spreads


def get_spy_last_price(ticker: Ticker) -> Optional[float]:
    """尽量取实时价格。"""
    for px in [ticker.last, ticker.marketPrice(), ticker.close, ticker.bid, ticker.ask]:
        if px is not None and not math.isnan(px):
            return float(px)
    return None


def close_leg(ib: IB, pos: Position, qty: int):
    """按腿下市价单，qty 为绝对值（合约张数）。"""
    from ib_insync import MarketOrder

    if qty <= 0:
        return
    action = "BUY" if pos.position < 0 else "SELL"
    order = MarketOrder(action, qty)
    trade = ib.placeOrder(pos.contract, order)
    logging.info("发送订单: %s %s x %s (orderId=%s)", action, pos.contract.localSymbol, qty, order.orderId)
    return trade


def reduce_spread_half(ib: IB, spread: SpreadPair) -> bool:
    """把该 spread 减半（按整张取整，至少 1 张）。"""
    current = spread.spread_size
    reduce_qty = current // 2 if current >= 2 else 1
    reduce_qty = min(reduce_qty, current)
    if reduce_qty <= 0:
        return False

    close_leg(ib, spread.short_leg, reduce_qty)
    close_leg(ib, spread.long_leg, reduce_qty)
    logging.warning(
        "触发减半: %s %s %s short=%.1f long=%.1f 中点=%.2f, 减仓=%s",
        spread.symbol,
        spread.right,
        spread.expiry,
        spread.short_strike,
        spread.long_strike,
        spread.mid_strike,
        reduce_qty,
    )
    return True


def close_spread_all(ib: IB, spread: SpreadPair) -> bool:
    """平掉该 spread 全部剩余仓位。"""
    qty = spread.spread_size
    if qty <= 0:
        return False

    close_leg(ib, spread.short_leg, qty)
    close_leg(ib, spread.long_leg, qty)
    logging.warning(
        "触发平仓: %s %s %s short=%.1f long=%.1f 中点=%.2f, 平仓=%s",
        spread.symbol,
        spread.right,
        spread.expiry,
        spread.short_strike,
        spread.long_strike,
        spread.mid_strike,
        qty,
    )
    return True


def monitor_and_manage(ib: IB, poll_seconds: float = 1.0):
    today_positions = load_today_spy_option_positions(ib)
    if not today_positions:
        logging.error("未找到当日到期的 SPY 期权持仓，退出。")
        return

    spreads = build_vertical_spreads(today_positions)
    if not spreads:
        logging.error("未能从持仓中匹配出 vertical spread（请检查是否为一一对应的双腿）。")
        return

    logging.info("识别到 %s 个 spread：", len(spreads))
    for i, sp in enumerate(spreads, start=1):
        logging.info(
            "%s) %s %s %s short=%.1f(%s) long=%.1f(%s) size=%s mid=%.2f",
            i,
            sp.symbol,
            sp.right,
            sp.expiry,
            sp.short_strike,
            sp.short_leg.position,
            sp.long_strike,
            sp.long_leg.position,
            sp.spread_size,
            sp.mid_strike,
        )

    # 跟踪每个spread阶段：0=未触发,1=已减半,2=已平仓
    stage: Dict[int, int] = {idx: 0 for idx in range(len(spreads))}

    spy = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(spy)
    ticker = ib.reqMktData(spy, "", False, False)

    logging.info("开始监控 SPY 实时价格...")

    while True:
        ib.sleep(poll_seconds)
        price = get_spy_last_price(ticker)
        if price is None:
            continue

        logging.info("SPY=%.4f", price)

        # 每次都刷新持仓，避免成交后状态不一致
        latest_positions = load_today_spy_option_positions(ib)
        latest_spreads = build_vertical_spreads(latest_positions)

        # 用 key 重新对齐 spread
        latest_map = {
            (
                sp.symbol,
                sp.right,
                sp.expiry,
                sp.short_leg.contract.conId,
                sp.long_leg.contract.conId,
            ): sp
            for sp in latest_spreads
        }

        for idx, old_sp in enumerate(spreads):
            key = (
                old_sp.symbol,
                old_sp.right,
                old_sp.expiry,
                old_sp.short_leg.contract.conId,
                old_sp.long_leg.contract.conId,
            )
            sp = latest_map.get(key)
            if not sp:
                # 已完全离场或结构变化
                stage[idx] = 2
                continue

            if sp.spread_size <= 0:
                stage[idx] = 2
                continue

            # 规则 1: 先看中间价 -> 直接平仓优先级更高
            if stage[idx] < 2 and price <= sp.mid_strike:
                if close_spread_all(ib, sp):
                    stage[idx] = 2
                continue

            # 规则 2: 到 short 腿执行价 -> 减半（只触发一次）
            if stage[idx] == 0 and price <= sp.short_strike:
                if reduce_spread_half(ib, sp):
                    stage[idx] = 1

        if all(v == 2 for v in stage.values()):
            logging.info("所有 spread 已完成平仓，监控结束。")
            return


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IB SPY spread 风控脚本")
    p.add_argument("--host", default="127.0.0.1", help="IB Gateway/TWS 地址")
    p.add_argument("--port", type=int, default=7497, help="端口，纸交易一般为 7497")
    p.add_argument("--client-id", type=int, default=77, help="IB client id")
    p.add_argument("--poll", type=float, default=1.0, help="价格轮询间隔（秒）")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ib = IB()

    def handle_stop(_signum, _frame):
        logging.warning("收到中断信号，准备退出...")
        try:
            ib.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    try:
        ib.connect(args.host, args.port, clientId=args.client_id)
        logging.info("已连接 IB: %s:%s clientId=%s", args.host, args.port, args.client_id)

        # 请求实时行情权限不足时，自动回落到延迟行情
        ib.reqMarketDataType(1)

        monitor_and_manage(ib, poll_seconds=args.poll)
    finally:
        if ib.isConnected():
            ib.disconnect()
            logging.info("IB 已断开。")


if __name__ == "__main__":
    util.patchAsyncio()
    main()
