from app.community.plugins.coin_price_chart_similarity_search.tools import TOOLS
from app.core.plugins import BasePlugin
from app.core.plugins.registry import ensure_plugin_registry
from app.core.plugins.tools import BaseTool

plugin_registry = ensure_plugin_registry()

@plugin_registry.register(name="coin_price_chart_similarity_search")
class CoinPriceChartSimilaritySearchPlugin(BasePlugin):
    name = "coin_price_chart_similarity_search"
    description = "Search coins similar to a given coin by price chart similarity."
    tools: type[list[BaseTool]] = TOOLS