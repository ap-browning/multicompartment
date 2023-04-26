using Plots

gr()
default()
default(
    fontfamily="Helvetica",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Helvetica",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Arial"
    )
end

## Colours
cols = [RGB(0/255, 122/255, 255/255); palette(:PuRd_6,rev=true)[1:6]]
grey = RGB(0.6,0.6,0.6)
grad = palette(:Oranges_9,rev=true)[[5,3,1]]
grad = palette(:Greens_8,rev=true)[[5,3,1]]
# 88
# G 86
# B 214