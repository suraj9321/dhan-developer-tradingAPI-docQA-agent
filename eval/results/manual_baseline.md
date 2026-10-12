# Manual check: `baseline` system

## 1. Which endpoint returns the option chain for an underlying?

The endpoint that returns the option chain for an underlying is `POST https://api.dhan.co/v2/optionchain` [1][4].

- [1] POST https://api.dhan.co/v2/optionchain  _(Indian Trading & Data API > Get Option Chain)_
- [4] Indian Trading & Data API > Option Chain  _(Indian Trading & Data API > Option Chain)_

`12085 ms · 962+31 tokens · $0.00023 · refused=False`

## 2. What values can productType take when placing an order?

The `productType` can take the following values: `CNC`, `INTRADAY`, `MARGIN`, `MTF` [1].

- [1] Developer Guides > Place Conditional Order  _(Developer Guides > Place Conditional Order)_

`1492 ms · 1345+32 tokens · $0.00031 · refused=False`

## 3. What does error code DH-904 mean?

Error code `DH-904` means "Rate Limit" and the message is "Too many requests — try throttling API calls" [2].

- [2] Annexure > Trading API Error  _(Annexure > Trading API Error)_

`8168 ms · 994+31 tokens · $0.00023 · refused=False`

## 4. How do I cancel an order in the Global Stocks API?

To cancel an order in the Global Stocks API, you can use the `DELETE` method with the endpoint `/globalstocks/orders/{order-id}` [1][2]. The `order-id` is a required path parameter that represents the Order ID of the order being cancelled [2]. An example request can be made using `curl -X DELETE https://api.dhan.co/v2/globalstocks/orders/{order-id} -H "access-token: <your-access-token>"` [2]. A successful cancellation will return a status code of `200` [2].

- [1] Global Stocks API > Orders  _(Global Stocks API > Orders)_
- [2] DELETE https://api.dhan.co/v2/globalstocks/orders/{order-id}  _(Global Stocks API > Cancel Order)_

`5552 ms · 1601+115 tokens · $0.00041 · refused=False`

## 5. What is the rate limit for order APIs per second?

The rate limit for Order APIs is 10 requests per second [2][3].

- [2] Developer Guides > Rate Limits  _(Developer Guides > Rate Limits)_
- [3] Indian Trading & Data API > DhanHQ API > Rate Limits  _(Indian Trading & Data API > DhanHQ API > Rate Limits)_

`974 ms · 1059+18 tokens · $0.00024 · refused=False`

## 6. Which header carries the access token on REST requests?

The `access-token` header carries the access token on REST requests [1][2].

- [1] Developer Guides > Getting Started > API Structure  _(Developer Guides > Getting Started > API Structure)_
- [2] Indian Trading & Data API > DhanHQ API > Structure  _(Indian Trading & Data API > DhanHQ API > Structure)_

`2444 ms · 735+19 tokens · $0.00017 · refused=False`

## 7. How do I set up a static IP for API access?

You can set up a static IP for API access by using the [Set IP API](/api/v2/authentication/set-ip) or by going to [web.dhan.co](https://web.dhan.co) → DhanHQ APIs section [1][2]. The endpoint to set a primary or secondary static IP is `POST https://api.dhan.co/v2/ip/setIP` [3].

- [1] Authentication > Setup Static IP  _(Authentication > Setup Static IP)_
- [2] Agent Skills > Configuration > Static IP Whitelisting  _(Agent Skills > Configuration > Static IP Whitelisting)_
- [3] POST https://api.dhan.co/v2/ip/setIP  _(Authentication > Set IP)_

`8841 ms · 1222+83 tokens · $0.00031 · refused=False`

## 8. What fields does the Get Holdings response include for Indian stocks?

The Get Holdings response for Indian stocks includes the following fields: `exchange`, `tradingSymbol`, `securityId`, `isin`, `totalQty`, `dpQty`, `t1Qty`, `mtf_t1_qty`, `mtf_qty`, `availableQty`, `collateralQty`, `avgCostPrice`, and `lastTradedPrice` [1].

- [1] GET https://api.dhan.co/v2/holdings  _(Indian Trading & Data API > Get Holdings)_

`1890 ms · 1273+77 tokens · $0.00032 · refused=False`

## 9. How do I exit all open positions at once?

You can exit all open positions at once by sending a `DELETE` request to the `https://api.dhan.co/v2/positions` endpoint [1]. This will exit all active positions and cancel all open orders for the current trading day, returning a `202 Accepted` response status code on successful completion [1]. Alternatively, the status code could be `200` indicating all positions have been exited [1].

- [1] DELETE https://api.dhan.co/v2/positions  _(Indian Trading & Data API > Exit All Positions)_

`2117 ms · 1198+85 tokens · $0.00031 · refused=False`

## 10. Does DhanHQ support trading cryptocurrency futures through the API?

I don't know based on the DhanHQ documentation.


`1140 ms · 1049+13 tokens · $0.00024 · refused=True`
