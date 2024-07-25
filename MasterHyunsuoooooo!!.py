import pandas as pd

# 산업안전보건관리비 계산기
def calculate_industrial_safety_health_management_cost(material_cost, supplied_material_cost, direct_labor_cost,
                                                       construction_type):
    # 총 비용 계산
    total_cost = material_cost + supplied_material_cost + direct_labor_cost

    # 계상기준비율 설정
    if construction_type == "일반건설공사(갑)":
        if total_cost < 500000000:
            rate = 0.0293
            base_cost = 0
        elif 500000000 <= total_cost < 5000000000:
            rate = 0.0186
            base_cost = 5349000
        else:
            rate = 0.0197
            base_cost = 0

    elif construction_type == "일반건설공사(을)":
        if total_cost < 500000000:
            rate = 0.0309
            base_cost = 0
        elif 500000000 <= total_cost < 5000000000:
            rate = 0.0199
            base_cost = 5499000
        else:
            rate = 0.021
            base_cost = 0

    elif construction_type == "중건설공사":
        if total_cost < 500000000:
            rate = 0.0343
            base_cost = 0
        elif 500000000 <= total_cost < 5000000000:
            rate = 0.0235
            base_cost = 5400000
        else:
            rate = 0.0244
            base_cost = 0

    elif construction_type == "철도-궤도신설공사":
        if total_cost < 500000000:
            rate = 0.0245
            base_cost = 0
        elif 500000000 <= total_cost < 5000000000:
            rate = 0.0157
            base_cost = 4411000
        else:
            rate = 0.0166
            base_cost = 0

    elif construction_type == "특수및기타건설공사":
        if total_cost < 500000000:
            rate = 0.0185
            base_cost = 0
        elif 500000000 <= total_cost < 5000000000:
            rate = 0.012
            base_cost = 3250000
        else:
            rate = 0.0127
            base_cost = 0

    else:
        raise ValueError("올바른 건설공사 유형을 선택하세요.")

    # 산업안전보건관리비 계산
    safety_health_management_cost = total_cost * rate + base_cost

    return safety_health_management_cost


values = []

# 프로그램 루프 시작
while True:
    print("\n건설공사 유형을 선택하세요: 일반건설공사(갑), 일반건설공사(을), 중건설공사, 철도-궤도신설공사, 특수및기타건설공사, end")
    construction_type = input("건설공사 유형을 입력하세요: ").strip()

    if construction_type.lower() == "end":
        print("프로그램을 종료합니다.")
        break

    try:
        material_cost = float(input("재료비를 입력하세요: "))
        supplied_material_cost = float(input("관급재료비를 입력하세요: "))
        direct_labor_cost = float(input("직접노무비를 입력하세요: "))

        # 산업안전보건관리비 계산
        safety_health_management_cost = calculate_industrial_safety_health_management_cost(
            material_cost, supplied_material_cost, direct_labor_cost, construction_type
        )

        print(f"산업안전보건관리비: {safety_health_management_cost:.2f} 원 입니다.")

    # 결과값 excel 저장 코드
        columns = ["공사종류", "재료비", "관급재료비", "직접노무비", "산업안전보건관리비"]
        values.append([construction_type, material_cost, supplied_material_cost, direct_labor_cost,
                   safety_health_management_cost])
        df = pd.DataFrame(values, columns=columns)
        print(df)
        df.to_excel("./result2.xlsx")

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")



