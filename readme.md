## AlphaStock

### Environment
* alpha_env: 按照文章复现，action采用0-1向量形式

### Executable
* alphastock_pg：简单的policy gradient, action形式为dim=stock_num的{0,1,-1}向量（代表买/卖/不动）
* alphastockd_pg：简单的policy gradient, action形式为[buy_id,sell_id]向量
* alphastock_ppo：ppo代码框架，尚不能跑，需要解决π(s|a)计算问题
