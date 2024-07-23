money = int(input('대상액을 입력하시오. :'))
if money < 500000000 :
   safety_money = 0.0293 * money

elif 500000000 <= money < 5000000000 :
    safety_money = 0.0186 * money + 5349000

elif money >= 5000000000 :
    safety_money = 0.0197 * money

print(int(safety_money))